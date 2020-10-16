import os
import argparse


def main(args):
    """ try different values for epsilon and the decay
        and the two parameter from per alpha and beta
    """
    lr = [0.1, 0.5, 0.01, 0.05, 0.001]
    freqs = [1,2,4,8,16]
    for freq in freqs:
        for lr_r in lr:
            for lr_q in lr:
                os.system(f'python3 ./main_table_iql.py \
                                            --lr_iql_q {lr_q} \
                                            --lr_iql_r {lr_r}  \
                                            --freq_q {freq}')


if __name__ == "__main__":
    print("hyper search")
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--env_num', type=int, default=1, help='enviroment')
    parser.add_argument('--num_parallel', type=int, default=0, help='enviroment')
    main(parser.parse_args())