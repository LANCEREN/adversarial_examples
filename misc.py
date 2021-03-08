def auto_select_gpu(mem_bound=500, utility_bound=5, gpus=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                    num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVCIES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5): # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert(len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

        if len(ideal_gpus) < num_gpu:
            print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')

    print("Setting GPU: {}".format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)

    return selected_gpus
