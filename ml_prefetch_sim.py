#!/usr/bin/env python3

import argparse
import os
import sys

default_results_dir = './results'
default_output_file = './stats.csv'
default_spec_instrs = 500
default_gap_instrs = 300
default_warmup_instrs = 0

default_base_binary = 'bin/perceptron-no-no-no-no-lru-1core'
default_prefetcher_binary = 'bin/perceptron-no-no-no-from_file-lru-1core'

help_str = {
'help': '''usage: {prog} command [<args>]

Available commands:
    build            Builds base and prefetcher ChampSim binaries
    run              Runs ChampSim on specified traces
    eval             Parses and computes metrics on simulation results
    help             Displays this help message. Command-specific help messages
                     can be displayed with `{prog} help command`
'''.format(prog=sys.argv[0]),

'build': '''usage: {prog} build [<target>]

Description:
    {prog} build [<target>]
        Builds <target> ChampSim binaries where <target> is one of:

            all            Builds both the base and prefetcher binaries [default]
            base           Builds just the base binary
            prefetcher     Builds just the prefetcher binary

        If <target> is unspecified, this will act as if `{prog} build all` was
        executed.

Notes:
    Barring updates to the GitHub repository, this will only need to be done once.
'''.format(prog=sys.argv[0]),

'run': '''usage: {prog} run <execution-trace> [--prefetch <prefetch-file>] [--no-base] [--results-dir <results-dir>]
                            [--num-instructions <num-instructions>] [--num-warmup-instructions <num-warmup-instructions>]

Description:
    {prog} run <execution-trace>
        Runs the base ChampSim binary on the specified execution trace.

Options:
    --prefetch <prefetch-file>
        Additionally runs the prefetcher ChampSim binary that issues prefetches
        according to the file.

    --no-base
        When specified with --prefetch <prefetch-file>, run only the prefetcher
        ChampSim binary on the specified execution trace without the base
        ChampSim binary.

    --results-dir <results-dir>
        Specifies what directory to save the ChampSim results file in. This
        defaults to `{default_results_dir}`.

    --num-instructions <num-instructions>
        Number of instructions to run the simulation for. Defaults to
        {default_spec_instrs}M instructions for the spec benchmarks and
        {default_gap_instrs}M instructions for the gap benchmarks.

    --num-warmup-instructions <num-warmup-instructions>
        Number of instructions to warm-up the simulator for. Defaults to
        {default_warmup_instrs}M instructions.
'''.format(prog=sys.argv[0], default_results_dir=default_results_dir,
    default_spec_instrs=default_spec_instrs, default_gap_instrs=default_gap_instrs,
    default_warmup_instrs=default_warmup_instrs),

'eval': '''usage: {prog} eval [--results-dir <results-dir>] [--output-file <output-file>]

Description:
    {prog} eval
        Runs the evaluation procedure on the ChampSim output found in the specified
        results directory and outputs a CSV at the specified output path.

Options:
    --results-dir <results-dir>
        Specifies what directory the ChampSim results files are in. This defaults
        to `{default_results_dir}`.

    --output-file <output-file>
        Specifies what file path to save the stats CSV data to. This defaults to
        `{default_output_file}`.

Note:
    To get stats comparing performance to a no-prefetcher baseline, it is necessary
    to have run the base ChampSim binary on the same execution trace.

    Without the base data, relative performance data comparing MPKI and IPC will
    not be available and the coverage statistic will only be approximate.
'''.format(prog=sys.argv[0], default_results_dir=default_results_dir, default_output_file=default_output_file),
}

def build_command():
    build = 'all'
    if len(sys.argv) > 2:
        if sys.argv[2] == 'all':
            build = 'all'
        elif sys.argv[2] == 'base':
            build = 'base'
        elif sys.argv[2] == 'prefetcher':
            build = 'prefetcher'
        else:
            print('Invalid build target')
            exit(-1)

    # Build base
    if build in ['all', 'base']:
        print('Building base ChampSim binary')
        os.system('./build_champsim.sh perceptron no no no no lru 1')

    # Build prefetcher
    if build in ['all', 'prefetcher']:
        print('Building prefetcher ChampSim binary')
        os.system('./build_champsim.sh perceptron no no no from_file lru 1')

def run_command():
    if len(sys.argv) < 3:
        print(help_str['run'])
        exit(-1)

    execution_trace = sys.argv[2]

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('--prefetch', default=None)
    parser.add_argument('--no-base', default=False, action='store_true')
    parser.add_argument('--results-dir', default=default_results_dir)
    parser.add_argument('--num-instructions', default=default_spec_instrs if execution_trace[0].isdigit() else default_gap_instrs)
    parser.add_argument('--num-warmup-instructions', default=default_warmup_instrs)

    args = parser.parse_args(sys.argv[3:])

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    if not args.no_base:
        if not os.path.exists(default_base_binary):
            print('Base ChampSim binary not found')
            exit(-1)

        cmd = '{binary} -warmup_instructions {warm}000000 -simulation_instructions {sim}000000 -traces {trace} > {results}/{base_trace}-{base_binary}.txt 2>&1'.format(
            binary=default_base_binary, warm=args.num_warmup_instructions, sim=args.num_instructions,
            trace=execution_trace, results=args.results_dir, base_trace=os.path.basename(execution_trace),
            base_binary=os.path.basename(default_base_binary))

        print('Running "' + cmd + '"')

        os.system(cmd)

    if args.prefetch is not None:
        if not os.path.exists(default_prefetcher_binary):
            print('Prefetcher ChampSim binary not found')
            exit(-1)

        cmd = '<{prefetch} {binary} -warmup_instructions {warm}000000 -simulation_instructions {sim}000000 -traces {trace} > {results}/{base_trace}-{base_binary}.txt 2>&1'.format(
            prefetch=args.prefetch, binary=default_prefetcher_binary, warm=args.num_warmup_instructions, sim=args.num_instructions,
            trace=execution_trace, results=args.results_dir, base_trace=os.path.basename(execution_trace),
            base_binary=os.path.basename(default_prefetcher_binary))

        print('Running "' + cmd + '"')

        os.system(cmd)

def read_file(path, cache_level='LLC'):
    expected_keys = ('ipc', 'total_miss', 'useful', 'useless', 'load_miss', 'rfo_miss', 'kilo_inst')
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if 'Finished CPU' in line:
                data['ipc'] = float(line.split()[9])
                data['kilo_inst'] = int(line.split()[4]) / 1000
            if cache_level not in line:
                continue
            line = line.strip()
            if 'LOAD' in line:
                data['load_miss'] = int(line.split()[-1])
            elif 'RFO' in line:
                data['rfo_miss'] = int(line.split()[-1])
            elif 'TOTAL' in line:
                data['total_miss'] = int(line.split()[-1])
            elif 'USEFUL' in line:
                data['useful'] = int(line.split()[-3])
                data['useless'] = int(line.split()[-1])

    if not all(key in data for key in expected_keys):
        return None

    return data

def compute_stats(trace, prefetch=None, base=None):
    if prefetch is None:
        return None

    pf_data = read_file(prefetch)

    useful, useless, ipc, load_miss, rfo_miss, kilo_inst = (
        pf_data['useful'], pf_data['useless'], pf_data['ipc'], pf_data['load_miss'], pf_data['rfo_miss'], pf_data['kilo_inst']
    )
    pf_total_miss = load_miss + rfo_miss + useful
    total_miss = pf_total_miss

    pf_mpki = (load_miss + rfo_miss) / kilo_inst

    if base is not None:
        b_data = read_file(base)
        b_total_miss, b_ipc = b_data['total_miss'], b_data['ipc']
        b_mpki = b_total_miss / kilo_inst

    if useful + useless == 0:
        acc = 'N/A'
    else:
        acc = str(useful / (useful + useless) * 100)
    if total_miss == 0:
        cov = 'N/A'
    else:
        cov = str(useful / total_miss * 100)
    if base is not None:
        mpki_improv = str((b_mpki - pf_mpki) / b_mpki * 100)
        ipc_improv = str((ipc - b_ipc) / b_ipc * 100)
    else:
        mpki_improv = 'N/A'
        ipc_improv = 'N/A'

    return '{trace},{acc},{cov},{mpki},{mpki_improv},{ipc},{ipc_improv}'.format(
        trace=trace, acc=acc, cov=cov, mpki=str(pf_mpki), mpki_improv=mpki_improv,
        ipc=str(ipc), ipc_improv=ipc_improv,
    )

def eval_command():
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('--results-dir', default=default_results_dir)
    parser.add_argument('--output-file', default=default_output_file)

    args = parser.parse_args(sys.argv[2:])

    traces = {}
    for fn in os.listdir(args.results_dir):
        trace = fn.split('-perceptron-')[0]
        if trace not in traces:
            traces[trace] = {}
        if 'from_file' in fn:
            traces[trace]['prefetch'] = os.path.join(args.results_dir, fn)
        else:
            traces[trace]['base'] = os.path.join(args.results_dir, fn)

    stats = ['Trace,Accuracy,Coverage,MPKI,MPKI_Improvement,IPC,IPC_Improvement']
    for trace in traces:
        trace_stats = compute_stats(trace, **traces[trace])
        if trace_stats is not None:
            stats.append(trace_stats)
    print('\n'.join(stats))

def help_command():
    # If one of the available help strings, print and exit successfully
    if len(sys.argv) > 2 and sys.argv[2] in help_str:
        print(help_str[sys.argv[2]])
        exit()
    # Otherwise, invalid subcommand, so print main help string and exit
    else:
        print(help_str['help'])
        exit(-1)

commands = {
    'build': build_command,
    'run': run_command,
    'eval': eval_command,
    'help': help_command,
}

def main():
    # If no subcommand specified or invalid subcommand, print main help string and exit
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(help_str['help'])
        exit(-1)

    # Run specified subcommand
    commands[sys.argv[1]]()

if __name__ == '__main__':
    main()
