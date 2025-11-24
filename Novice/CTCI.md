# Databases
- Inner join: Result contains only data which has a match in both tables.
- Outer join: Always contains results of inner join plus records that only match in one table.
- Left (outer) join: Contains all records from the left table plus the matching records in the right table.
- Right (outer) join: Contains all records from the right table plus the matching records in the left table.
- Full outer join: Contains all records from both tables.
- Denormalisation: Process of attempting to optimise performance by adding redundant data or grouping data.
- SELECT TOP x PERCENT column_name: Selects the top x percent from column name.

# Low Level
- Virtual memory: Computer system technique which gives an application program the impression that it has contiguous working memory (an address space), while in fact it may be physically fragmented and may even overflow to disk storage.
- Page fault: A page is a fixed-length block of memory used as a unit of transfer between physical memory and external storage like a disk.
    - A page fault is an interrupt (or exception) to the software raised by the hardware, when a program accesses a page that is mapped in address space but not loaded in physical memory.
- Thrashing: Term used to describe a degenerate situation on a computer where increasing resources are used to do a decreasing amount of work.
    - **Expand on this**

# Threads and Locks
- Process: Instance of a program in execution. Each process is an independent entity to which sytstem resources (CPU time, memory) are allocated and each process is exeucted in a seperate address space.
- Thread: Uses same stack space of a process, and a process can have multiple threads.
    - Multiple threads share parts of their state.
    - Multiple threads can read and write the same memory, though each thread still has its own registers and stack.
- Lock (Mutex): Simple tool used to prevent two or more threads from accessing the same piece of data at the same time. Almost like a key to a shared resource.