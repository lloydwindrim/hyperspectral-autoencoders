import sys
import time


def reporthook(count, block_size, total_size):
    '''
    Function to display download progress.

    From blog:
    https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
    '''
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


