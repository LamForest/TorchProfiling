import torch

STACK = []

class TagBackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        ctx.mark_non_differentiable(*[arg for arg in args if not arg.requires_grad])
        return args

    @staticmethod
    def backward(ctx, *args):
        return args

def apply_hook(args, fn):
    tensors = []
    tensors_idx = []
    requires_grad = False
    
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            tensors.append(arg)
            tensors_idx.append(i)
            requires_grad |= arg.requires_grad

    if not (requires_grad and torch.is_grad_enabled()):
        return args
    
    new_tensors = TagBackwardHook.apply(*tensors)

    
    assert len(new_tensors) > 0
    grad_fns = [t.grad_fn for t in new_tensors if t.grad_fn is not None and t.grad_fn.name() == "TagBackwardHookBackward"]
    
    print(f"{grad_fns=}")
    
    grad_fns[0].register_hook(fn)

    arg_list = list(args)
    for idx, val in zip(tensors_idx, new_tensors):
        arg_list[idx] = val
    return tuple(arg_list)

    
def tag_push(*args, tag:str):
    STACK.append(tag)
    args = apply_hook(args, fn=lambda *_a, **_b: print(f'[END BACKWARD]: {tag}')) #[BEGIN FORWARD]
    print(f"[BEGIN_FORWARD]: {tag}")
    if len(args) == 1: #简单粗暴，先这么用吧
        return args[0]
    return args

def tag_pop(ret, tag=None):
    #如果用户传入tag，那么验证是否期望的tag与栈顶的tag一致
    if tag is not None:
        assert tag == STACK[-1]
    tag = STACK.pop()
    
    is_tuple = True
    if not isinstance(ret, tuple):
        ret = (ret,)
        is_tuple = False
    ret = apply_hook(ret, fn=lambda *_a, **_b: print(f'[BEGIN BACKWARD]: {tag}'))
    print(f"[END FORWARD]: {tag}")
    
    if not is_tuple:
        ret = ret[0]
    return ret


def tag(tag="default"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if kwargs is None:
                kwargs = {}
            assert len(kwargs) == 0, f"暂不支持kwargs hook"
            args = tag_push(*args, tag=tag)
            if not isinstance(args, tuple):
                args = (args,)
            ret = func(*args, **kwargs)
            
            ret = tag_pop(ret, tag=tag)
            return ret
        return wrapper
    return decorator

