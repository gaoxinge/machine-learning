## 第3章 目标文件里有什么

### ELF

- COFF (Common File Format)
  - linux: ELF (Executable Linkable Format)
  - windows: PE/COFF (Portable Executable)
- ELF
  - 可重定位文件: .o
  - 可执行文件: .out
  - 共享目标文件: .so
- PE/COFF
  - 可重定位文件: .obj
  - 可执行文件: .exe
  - 共享目标文件: .dll

### SimpleSection.c

```c
int printf(const char* format, ...);
int global_init_var = 84;
int global_uninit_var;

void func1(int i) {
    printf("%d\n", i);
}

int main() {
    static int static_var = 85;
    static int static_var2;

    int a = 1;
    int b;
    
    func1(static_var + static_var2 + a + b);
   
    return 0;
}
```

```
$ gcc -c SimpleSection.c -o SimpleSection.o
```

### 目标文件的结构

```
$ readelf -S SimpleSection.o
```

```
共有 13 个节头，从偏移量 0x430 开始：

节头：
  [号] 名称              类型             地址              偏移量
       大小              全体大小          旗标   链接   信息   对齐
  [ 0]                   NULL             0000000000000000  00000000
       0000000000000000  0000000000000000           0     0     0
  [ 1] .text             PROGBITS         0000000000000000  00000040
       0000000000000057  0000000000000000  AX       0     0     1
  [ 2] .rela.text        RELA             0000000000000000  00000320
       0000000000000078  0000000000000018   I      11     1     8
  [ 3] .data             PROGBITS         0000000000000000  00000098
       0000000000000008  0000000000000000  WA       0     0     4
  [ 4] .bss              NOBITS           0000000000000000  000000a0
       0000000000000004  0000000000000000  WA       0     0     4
  [ 5] .rodata           PROGBITS         0000000000000000  000000a0
       0000000000000004  0000000000000000   A       0     0     1
  [ 6] .comment          PROGBITS         0000000000000000  000000a4
       0000000000000036  0000000000000001  MS       0     0     1
  [ 7] .note.GNU-stack   PROGBITS         0000000000000000  000000da
       0000000000000000  0000000000000000           0     0     1
  [ 8] .eh_frame         PROGBITS         0000000000000000  000000e0
       0000000000000058  0000000000000000   A       0     0     8
  [ 9] .rela.eh_frame    RELA             0000000000000000  00000398
       0000000000000030  0000000000000018   I      11     8     8
  [10] .shstrtab         STRTAB           0000000000000000  000003c8
       0000000000000061  0000000000000000           0     0     1
  [11] .symtab           SYMTAB           0000000000000000  00000138
       0000000000000180  0000000000000018          12    11     8
  [12] .strtab           STRTAB           0000000000000000  000002b8
       0000000000000066  0000000000000000           0     0     1
Key to Flags:
  W (write), A (alloc), X (execute), M (merge), S (strings), l (large)
  I (info), L (link order), G (group), T (TLS), E (exclude), x (unknown)
  O (extra OS processing required) o (OS specific), p (processor specific)
```

```
-----------------
|.strtab         |
-----------------
|.rel.text       |
-----------------
|.symtab         |
-----------------
|.shstrtab       |
-----------------
|.comment        |
-----------------
|.bss            |
-----------------
|.data           |
-----------------
|.rodata         |
-----------------
|.text           |
-----------------
|ELF Header      |
-----------------
```

```
$ objdump -h SimpleSection.o
```

```

SimpleSection.o：     文件格式 elf64-x86-64

节：
Idx Name          Size      VMA               LMA               File off  Algn
  0 .text         00000057  0000000000000000  0000000000000000  00000040  2**0
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  1 .data         00000008  0000000000000000  0000000000000000  00000098  2**2
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000004  0000000000000000  0000000000000000  000000a0  2**2
                  ALLOC
  3 .rodata       00000004  0000000000000000  0000000000000000  000000a0  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  4 .comment      00000036  0000000000000000  0000000000000000  000000a4  2**0
                  CONTENTS, READONLY
  5 .note.GNU-stack 00000000  0000000000000000  0000000000000000  000000da  2**0
                  CONTENTS, READONLY
  6 .eh_frame     00000058  0000000000000000  0000000000000000  000000e0  2**3
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, DATA
```

```
-----------------
|.comment        |
-----------------
|.bss            |
-----------------
|.data           |
-----------------
|.rodata         |
-----------------
|.text           |
-----------------
|ELF Header      |
-----------------
```


```
-----------------
|.strtab         |
-----------------
|.line           |
-----------------
|.debug          |
-----------------
|.rel.data       |
-----------------
|.rel.text       |
-----------------
|.symtab         |
-----------------
|Section Table   |
-----------------
|.shstrtab       |
-----------------
|.comment        |
-----------------
|.bss            |
-----------------
|.data           |
-----------------
|.rodata         |
-----------------
|.text           |
-----------------
|ELF Header      |
-----------------
```

### ELF Header

```
$ readelf -h SimpleSection.o
```

```
ELF 头：
  Magic：   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
  类别:                              ELF64
  数据:                              2 补码，小端序 (little endian)
  版本:                              1 (current)
  OS/ABI:                            UNIX - System V
  ABI 版本:                          0
  类型:                              REL (可重定位文件)
  系统架构:                          Advanced Micro Devices X86-64
  版本:                              0x1
  入口点地址：               0x0
  程序头起点：          0 (bytes into file)
  Start of section headers:          1072 (bytes into file)
  标志：             0x0
  本头的大小：       64 (字节)
  程序头大小：       0 (字节)
  Number of program headers:         0
  节头大小：         64 (字节)
  节头数量：         13
  字符串表索引节头： 10
```

### .text

```
$ objdump -d SimpleSection.o
```

```

SimpleSection.o：     文件格式 elf64-x86-64


Disassembly of section .text:

0000000000000000 <func1>:
   0:	55                   	push   %rbp
   1:	48 89 e5             	mov    %rsp,%rbp
   4:	48 83 ec 10          	sub    $0x10,%rsp
   8:	89 7d fc             	mov    %edi,-0x4(%rbp)
   b:	8b 45 fc             	mov    -0x4(%rbp),%eax
   e:	89 c6                	mov    %eax,%esi
  10:	bf 00 00 00 00       	mov    $0x0,%edi
  15:	b8 00 00 00 00       	mov    $0x0,%eax
  1a:	e8 00 00 00 00       	callq  1f <func1+0x1f>
  1f:	90                   	nop
  20:	c9                   	leaveq 
  21:	c3                   	retq   

0000000000000022 <main>:
  22:	55                   	push   %rbp
  23:	48 89 e5             	mov    %rsp,%rbp
  26:	48 83 ec 10          	sub    $0x10,%rsp
  2a:	c7 45 f8 01 00 00 00 	movl   $0x1,-0x8(%rbp)
  31:	8b 15 00 00 00 00    	mov    0x0(%rip),%edx        # 37 <main+0x15>
  37:	8b 05 00 00 00 00    	mov    0x0(%rip),%eax        # 3d <main+0x1b>
  3d:	01 c2                	add    %eax,%edx
  3f:	8b 45 f8             	mov    -0x8(%rbp),%eax
  42:	01 c2                	add    %eax,%edx
  44:	8b 45 fc             	mov    -0x4(%rbp),%eax
  47:	01 d0                	add    %edx,%eax
  49:	89 c7                	mov    %eax,%edi
  4b:	e8 00 00 00 00       	callq  50 <main+0x2e>
  50:	b8 00 00 00 00       	mov    $0x0,%eax
  55:	c9                   	leaveq 
  56:	c3                   	retq   
```

### .rodata和.data

```
$ objdump -x -s SImpleSection.o
```

- 初始化的全局变量和局部静态变量存放于.data：`global_init_var`和`static_var`
- const修饰的变量和字符串常量存放于.rodata

### .bss

```
$ objdump -x -s SImpleSection.o
```

- 未初始化的全局变量和局部静态变量存放于.bss：`global_uninit_var`和`static_var2`

### Section Table

```
$ readelf -S SimpleSection.o
```

- 段表（Section Table）：用于存放section的信息

### .shstrtab和.strtab

- 段表字符串表（Section Header String Table）
- 字符串表（String Table）

### .symtab

```
$ objdump -x SimpleSection
```

```
SYMBOL TABLE:
0000000000000000 l    df *ABS*	0000000000000000 SimpleSection.c
0000000000000000 l    d  .text	0000000000000000 .text
0000000000000000 l    d  .data	0000000000000000 .data
0000000000000000 l    d  .bss	0000000000000000 .bss
0000000000000000 l    d  .rodata	0000000000000000 .rodata
0000000000000004 l     O .data	0000000000000004 static_var.1839
0000000000000000 l     O .bss	0000000000000004 static_var2.1840
0000000000000000 l    d  .note.GNU-stack	0000000000000000 .note.GNU-stack
0000000000000000 l    d  .eh_frame	0000000000000000 .eh_frame
0000000000000000 l    d  .comment	0000000000000000 .comment
0000000000000000 g     O .data	0000000000000004 global_init_var
0000000000000004       O *COM*	0000000000000004 global_uninit_var
0000000000000000 g     F .text	0000000000000022 func1
0000000000000000         *UND*	0000000000000000 printf
0000000000000022 g     F .text	0000000000000035 main
```

```
$ readelf -s SimpleSection.o
```

```
Symbol table '.symtab' contains 16 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS SimpleSection.c
     2: 0000000000000000     0 SECTION LOCAL  DEFAULT    1 
     3: 0000000000000000     0 SECTION LOCAL  DEFAULT    3 
     4: 0000000000000000     0 SECTION LOCAL  DEFAULT    4 
     5: 0000000000000000     0 SECTION LOCAL  DEFAULT    5 
     6: 0000000000000004     4 OBJECT  LOCAL  DEFAULT    3 static_var.1839
     7: 0000000000000000     4 OBJECT  LOCAL  DEFAULT    4 static_var2.1840
     8: 0000000000000000     0 SECTION LOCAL  DEFAULT    7 
     9: 0000000000000000     0 SECTION LOCAL  DEFAULT    8 
    10: 0000000000000000     0 SECTION LOCAL  DEFAULT    6 
    11: 0000000000000000     4 OBJECT  GLOBAL DEFAULT    3 global_init_var
    12: 0000000000000004     4 OBJECT  GLOBAL DEFAULT  COM global_uninit_var
    13: 0000000000000000    34 FUNC    GLOBAL DEFAULT    1 func1
    14: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND printf
    15: 0000000000000022    53 FUNC    GLOBAL DEFAULT    1 main
```

```
$ nm SimpleSection.o
```

```
0000000000000000 T func1
0000000000000000 D global_init_var
0000000000000004 C global_uninit_var
0000000000000022 T main
                 U printf
0000000000000004 d static_var.1839
0000000000000000 b static_var2.1840
```

- 符号表（Symbol Table）

### .rel.text和.rel.data

- 重定位表（Relocation Table）
