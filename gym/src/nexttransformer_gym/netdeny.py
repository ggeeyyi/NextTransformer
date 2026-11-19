"""Refuse new network sockets for a worker without a sandbox (isolation `none`, network `deny`).

An unprivileged seccomp filter (no_new_privs): it works where a machine has neither user namespaces nor Landlock.
socket(2) for the denied address families fails with EACCES; sockets opened before the filter, and pipes, keep
working. `python -m nexttransformer_gym.netdeny [--unix] -- COMMAND...` runs COMMAND under the filter (`--unix` also
denies Unix sockets, as the landlock backend does for CPU workers; CUDA needs them).
"""

import os
import sys

AF_UNIX, AF_INET, AF_INET6 = 1, 2, 10
SOCKET_SYSCALL = {"x86_64": (0xC000003E, 41), "aarch64": (0xC00000B7, 198)}
SECCOMP_SYSCALL = {"x86_64": 317, "aarch64": 277}


def deny_sockets(domains, all_threads=False):
    """Refuse socket(2) for `domains` with EACCES, for this process and its children.

    With `all_threads` the filter binds every thread of the process (SECCOMP_FILTER_FLAG_TSYNC), including threads
    started before it (a process group's own threads, for example)."""
    import ctypes
    import errno
    import platform

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint16),
            ("jt", ctypes.c_uint8),
            ("jf", ctypes.c_uint8),
            ("k", ctypes.c_uint32),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(Filter))]

    arch, socket_nr = SOCKET_SYSCALL[platform.machine()]
    load, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06
    n = len(domains)
    program = [
        (load, 0, 0, 4),
        (jeq, 1, 0, arch),
        (ret, 0, 0, 0x80000000),  # other architectures: kill
        (load, 0, 0, 0),
        (jge, 3 + n, 0, 0x40000000),  # x32 system calls: refuse
        (jeq, 0, 1 + n, socket_nr),
        (load, 0, 0, 16),  # socket(2): its domain
        *[(jeq, n - i, 0, domain) for i, domain in enumerate(domains)],
        (ret, 0, 0, 0x7FFF0000),
        (ret, 0, 0, 0x00050000 | errno.EACCES),
    ]
    filters = (Filter * len(program))(*(Filter(*f) for f in program))
    prog = Program(len(program), filters)
    libc = ctypes.CDLL(None, use_errno=True)

    def install():
        if all_threads:
            # seccomp(SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_TSYNC, prog); a thread id means one refused.
            return libc.syscall(
                SECCOMP_SYSCALL[platform.machine()], 1, 1, ctypes.byref(prog)
            )
        # prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, prog): this thread and its descendants.
        return libc.prctl(22, 2, ctypes.byref(prog), 0, 0)

    for call in (lambda: libc.prctl(38, 1, 0, 0, 0), install):  # PR_SET_NO_NEW_PRIVS
        if call() != 0:
            raise OSError(ctypes.get_errno(), "seccomp filter refused")


def command(python, worker, unix=False):
    """`worker` under the filter (IPv4 and IPv6 always, Unix sockets with `unix`); its environment must import
    this package, as a worker's does."""
    return [
        python,
        "-m",
        "nexttransformer_gym.netdeny",
        *(["--unix"] if unix else []),
        "--",
        *worker,
    ]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--")
    domains = [AF_INET, AF_INET6] + ([AF_UNIX] if "--unix" in argv[:split] else [])
    deny_sockets(domains)
    os.execvp(argv[split + 1], argv[split + 1 :])


if __name__ == "__main__":
    main()
