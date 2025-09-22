import jittor as jt
from contextlib import suppress
from contextlib import contextmanager



@contextmanager
def dummy_autocast(dtype=None):

    old_dtype = jt.float32 if dtype is None else dtype
    try:
        if dtype == jt.float16:
            jt.flags.auto_mixed_precision_level = 2
        elif dtype == jt.bfloat16:
            jt.flags.auto_mixed_precision_level = 3
        else:
            jt.flags.auto_mixed_precision_level = 0
        yield
    finally:

        jt.flags.auto_mixed_precision_level = 0


def get_autocast(precision):
    if precision == 'amp':
        return lambda: dummy_autocast(jt.float16)  
    elif precision in ('amp_bfloat16', 'amp_bf16'):
        return lambda: dummy_autocast(jt.bfloat16)  
    else:
        return suppress 
