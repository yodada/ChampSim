#!/bin/python3

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', help='Path to ChampSim results file')
    parser.add_argument('--cache-level', default='LLC', choices=('L2', 'LLC'), help='Cache level to compute stats for (default: %(default)s)')

    return parser.parse_args()


def main(args=None):
    print(args)
    with open(args.results_file, 'r') as f:
        for line in f:
            if 'Finished CPU' in line:
                ipc = float(line.split()[9])
            if args.cache_level not in line:
                continue
            line = line.strip()
            if 'TOTAL' in line:
                total_miss = int(line.split()[-1])
            elif 'USEFUL' in line:
                useful = int(line.split()[-3])
                useless = int(line.split()[-1])
                break

    if useful + useless == 0:
        print('Accuracy: N/A [All prefetches were merged and were not useful or useless]')
    else:
        print('Accuracy:', useful / (useful + useless))
    if total_miss == 0:
        print('Coverage: N/A [No misses. Did you run this simulation for long enough?]')
    else:
        print('Coverage:', useful / total_miss)
    print('IPC:', ipc)

if __name__ == '__main__':
    main(args=get_args())
