#include <cassert>
#include <cstring>

#include "config.h"

#ifdef HAVE_HIREDIS
#include <hiredis/hiredis.h>
#endif

#include "smtgcc.h"

using namespace std::string_literals;
namespace smtgcc {
namespace {

// The xxhash XXH64 algorithm as specified in:
//   https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
struct Hash
{
  template <typename T>
  void add(const T& data)
  {
    uint64_t stripe_size = data_size % sizeof(stripe);
    char *src = (char *)&data;
    uint64_t size = sizeof(data);
    while (size > 0)
      {
	uint64_t len = std::min(size, sizeof(stripe) - stripe_size);
	memcpy((char *)&stripe + stripe_size, src, len);
	src += len;
	stripe_size += len;
	size -= len;
	if (stripe_size == sizeof(stripe))
	  {
	    for (int i = 0; i < 4; i++)
	      acc[i] = round(acc[i], stripe[i]);
	    stripe_size = 0;
	  }
      }
    data_size += sizeof(data);
  }

  std::string finish()
  {
    uint64_t a;
    if (data_size < sizeof(stripe))
      a = prime5;
    else
      {
	a = rot(acc[0], 1);
	a += rot(acc[1], 7);
	a += rot(acc[2], 12);
	a += rot(acc[3], 18);
	for (int i = 0; i < 4; i++)
	  a = (a ^ round(0, acc[i])) * prime1 + prime4;
      }

    a = a + data_size;

    if (uint64_t remaining = data_size % sizeof(stripe); remaining != 0)
      {
	char *src = (char *)&stripe;
	while (remaining >= 8)
	  {
	    uint64_t lane;
	    memcpy(&lane, src, 8);
	    src += 8;
	    remaining -= 8;
	    a = rot(a ^ round(0, lane), 27) * prime1 + prime4;
	  }
	if (remaining >= 4)
	  {
	    uint32_t lane;
	    memcpy(&lane, src, 4);
	    src += 4;
	    remaining -= 4;
	    a = a ^ (lane * prime1);
	    a = rot(a, 23) * prime2 + prime3;
	  }
	while (remaining >= 1)
	  {
	    uint8_t lane;
	    memcpy(&lane, src, 1);
	    src += 1;
	    remaining -= 1;
	    a = a ^ (lane * prime5);
	    a = rot(a, 11) * prime1;
	  }
      }

    a = (a ^ (a >> 33)) * prime2;
    a = (a ^ (a >> 29)) * prime3;
    a = a ^ (a >> 32);

    char str[16 + 1];
    char *p = str;
    for (int i = 15; i >= 0; i--)
      *p++ = hex_char((a >> (i * 4)) & 0xf);
    *p++ = 0;

    return std::string(str);
  }

private:
  char hex_char(int x)
  {
    if (x < 10)
      return '0' + x;
    else
      return 'a' + (x - 10);
  }

  uint64_t rot(uint64_t x, uint64_t n)
  {
    return (x << n) | x >> (64 - n);
  }

  uint64_t round(uint64_t acc, uint64_t lane)
  {
    return rot(acc + (lane * prime2), 31) * prime1;
  }

  const uint64_t prime1 = 0x9E3779B185EBCA87;
  const uint64_t prime2 = 0xC2B2AE3D27D4EB4F;
  const uint64_t prime3 = 0x165667B19E3779F9;
  const uint64_t prime4 = 0x85EBCA77C2B2AE63;
  const uint64_t prime5 = 0x27D4EB2F165667C5;

  uint64_t data_size = 0;
  uint64_t acc[4] = {prime1 + prime2, prime2, 0, -prime1};
  uint64_t stripe[4];
};

} // end anonymous namespace

Cache::Cache(Function *func)
{
  if (config.redis_cache)
    key = hash(func);
}

std::string Cache::hash(Function *func)
{
  Hash h;

  // Add a version to the hash.
  //
  // We should try to remember to update the version when the implementation
  // changes in a way that invalidates the cache. In practice, this only
  // applies when the SMT code is updated or when inst_info is reordered.
  // Adding or removing inst_info values does not require the version to be
  // updated, as we instead add the number of inst_info as a separate value.
  const uint32_t version = 0;
  h.add(version);
  const uint32_t nof_inst_info = inst_info.size();
  h.add(nof_inst_info);

  // We add the SMT timeout/memory_limit/smt_solver to prevent the cache
  // from returning a timeout status from a previous run when we try again
  // with a larger timeout or different SMT solver.
  h.add(config.timeout);
  h.add(config.memory_limit);
  h.add(config.smt_solver);

  // Module config
  h.add(func->module->ptr_bits);
  h.add(func->module->ptr_id_bits);
  h.add(func->module->ptr_offset_bits);

  // Hash the function.
  for (Basic_block *bb : func->bbs)
    {
      for (auto phi : bb->phis)
	{
	  assert(phi->has_lhs());
	  h.add(phi->op);
	  h.add(phi->bitsize);
	  h.add(phi->id);
	  for (auto [arg_inst, arg_bb] : phi->phi_args)
	    {
	      h.add(arg_inst->id);
	      h.add(arg_bb->id);
	    }
	}
      for (Inst *inst = bb->first_inst; inst; inst = inst->next)
	{
	  h.add(inst->op);
	  if (inst->has_lhs())
	    h.add(inst->id);
	  switch (inst->iclass())
	    {
	    case Inst_class::iunary:
	    case Inst_class::funary:
	    case Inst_class::ibinary:
	    case Inst_class::fbinary:
	    case Inst_class::icomparison:
	    case Inst_class::fcomparison:
	    case Inst_class::conv:
	    case Inst_class::ternary:
	    case Inst_class::ls_unary:
	    case Inst_class::ls_binary:
	    case Inst_class::ls_ternary:
	    case Inst_class::mem_nullary:
	    case Inst_class::mem_ternary:
	    case Inst_class::reg_unary:
	    case Inst_class::reg_binary:
	    case Inst_class::solver_unary:
	    case Inst_class::solver_binary:
	    case Inst_class::solver_ternary:
	      for (uint16_t i = 0; i < inst->nof_args; i++)
		h.add(inst->args[i]->id);
	      break;
	    case Inst_class::special:
	      if (inst->op == Op::BR)
		{
		  if (inst->nof_args)
		    {
		      assert(inst->nof_args == 1);
		      h.add(inst->args[0]->id);
		      h.add(inst->u.br3.true_bb->id);
		      h.add(inst->u.br3.false_bb->id);
		    }
		  else
		    h.add(inst->u.br1.dest_bb->id);
		}
	      else if (inst->op == Op::RET)
		{
		  for (uint16_t i = 0; i < inst->nof_args; i++)
		    h.add(inst->args[i]->id);
		}
	      else if (inst->op == Op::VALUE)
		{
		  h.add(inst->bitsize);
		  h.add(inst->u.value.value);
		}
	      else
		assert(0);
	    }
	}
    }

  return h.finish();
}

std::optional<Solver_result> Cache::get()
{
#ifdef HAVE_HIREDIS
  if (config.redis_cache)
    {
      redisContext *ctx = redisConnect("127.0.0.1", 6379);
      if (!ctx)
	{
	  fprintf(stderr, "SMTGCC: Cannot allocate Redis context\n");
	  return {};
	}
      if (ctx->err)
	{
	  fprintf(stderr, "SMTGCC: Redis error: %s\n", ctx->errstr);
	  redisFree(ctx);
	  return {};
	}

      redisReply *reply =
	(redisReply *)redisCommand(ctx, "GET %s", key.c_str());
      if (!reply)
	{
	  fprintf(stderr, "SMTGCC: Redis error: %s\n", ctx->errstr);
	  redisFree(ctx);
	  return {};
	}
      if (reply->type == REDIS_REPLY_NIL)
	{
	  freeReplyObject(reply);
	  redisFree(ctx);
	  return {};
	}

      Result_status status;
      switch (reply->str[0])
	{
	case 'c':
	  status = Result_status::correct;
	  break;
	case 'i':
	  status = Result_status::incorrect;
	  break;
	case 'u':
	  status = Result_status::unknown;
	  break;
	default:
	  throw Not_implemented("Unhandled value read from database");
	}
      std::optional<std::string> message;
      if (reply->str[1])
	message = std::string(&reply->str[1]);

      freeReplyObject(reply);
      redisFree(ctx);

      return Solver_result(status, message);
    }
#endif

  return {};
}

void Cache::set(Solver_result result)
{
#ifdef HAVE_HIREDIS
  if (config.redis_cache)
    {
      redisContext *ctx = redisConnect("127.0.0.1", 6379);
      if (!ctx)
	{
	  fprintf(stderr, "SMTGCC: Cannot allocate Redis context\n");
	  return;
	}
      if (ctx->err)
	{
	  fprintf(stderr, "SMTGCC: Redis error: %s\n", ctx->errstr);
	  redisFree(ctx);
	  return;
	}

      std::string value;
      switch (result.status)
	{
	case Result_status::correct:
	  value = 'c';
	  break;
	case Result_status::incorrect:
	  value = 'i';
	  break;
	case Result_status::unknown:
	  value = 'u';
	  break;
	}
      if (result.message)
	value = value + *result.message;

      redisReply *reply = (redisReply *)redisCommand(ctx, "SET %s %s",
						     key.c_str(),
						     value.c_str());
      if (!reply || reply->type == REDIS_REPLY_ERROR)
	fprintf(stderr, "SMTGCC: Redis error: %s\n", ctx->errstr);
      if (reply)
	freeReplyObject(reply);
      redisFree(ctx);
    }
#endif
}

} // end namespace smtgcc
