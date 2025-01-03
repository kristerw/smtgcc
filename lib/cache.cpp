#include <cassert>

#include "config.h"

#ifdef HAVE_HIREDIS
#include <hiredis/hiredis.h>
#endif

#include "md5.h"
#include "smtgcc.h"

using namespace std::string_literals;
namespace smtgcc {
namespace {

struct Hash
{
  Hash()
  {
    md5_init_ctx(&ctx);
  }

  template <typename T>
  void add(const T& a)
  {
    md5_process_bytes(&a, sizeof(a), &ctx);
  }

  std::string finish()
  {
    uint8_t result[16];
    md5_finish_ctx(&ctx, result);

    char str[2 * 16 + 1];
    char *p = str;
    for (int i = 0; i < 16; i++)
      {
	*p++ = hex_char((result[i] & 0xf0) >> 4);
	*p++ = hex_char(result[i] & 0x0f);
      }
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

  md5_ctx ctx;
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

  // We add the SMT timeout/memory_limit to prevent the cache from returning
  // a time out status from a previous run when we try again with a larger
  // timeout.
  h.add(config.timeout);
  h.add(config.memory_limit);

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
	    case Inst_class::nullary:
	    case Inst_class::iunary:
	    case Inst_class::funary:
	    case Inst_class::ibinary:
	    case Inst_class::fbinary:
	    case Inst_class::icomparison:
	    case Inst_class::fcomparison:
	    case Inst_class::conv:
	    case Inst_class::ternary:
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
