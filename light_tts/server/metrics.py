import time

def histogram_timer(histogram):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time() 
            result = func(*args, **kwargs)  
            elapsed_time = time.time() - start_time  
            histogram.observe(elapsed_time) 
            return result
        return wrapper
    return decorator
