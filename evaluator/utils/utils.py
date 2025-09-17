from evaluator.eval_spec import VERBOSE


def print_verbose(message):
    if VERBOSE:
        print(message)


def print_iterable_verbose(label, iterable):
    print_verbose(label)
    for item in iterable:
        print_verbose(item)
