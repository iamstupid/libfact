.text
.p2align 4
.globl zfactor_fixint_sqr_diag_addlsh1_x86_64
.type zfactor_fixint_sqr_diag_addlsh1_x86_64,@function
zfactor_fixint_sqr_diag_addlsh1_x86_64:
    pushq %rbx
    pushq %r12
    pushq %r13

    movq %rdx, %r12
    movq %rcx, %r13

    movq (%r12), %rax
    mulq %rax
    movq %rax, (%rdi)
    movq %rdx, %r8

    addq $8, %rdi
    addq $8, %r12
    decq %r13
    jz .Ldone

    xorq %r9, %r9
    xorq %r10, %r10

.Lloop:
    movq (%rsi), %rax
    movq %rax, %r11
    shrq $63, %r11
    addq %rax, %rax
    orq %r10, %rax
    xorq %rbx, %rbx
    addq %r8, %rax
    adcq $0, %rbx
    addq %r9, %rax
    adcq $0, %rbx
    movq %rax, (%rdi)
    movq %rbx, %r9
    movq %r11, %r10

    movq (%r12), %rax
    mulq %rax

    movq 8(%rsi), %rbx
    movq %rbx, %r11
    shrq $63, %r11
    addq %rbx, %rbx
    orq %r10, %rbx
    xorq %r10, %r10
    addq %rax, %rbx
    adcq $0, %r10
    addq %r9, %rbx
    adcq $0, %r10
    movq %rbx, 8(%rdi)
    movq %rdx, %r8
    movq %r10, %r9
    movq %r11, %r10

    addq $16, %rsi
    addq $16, %rdi
    addq $8, %r12
    decq %r13
    jnz .Lloop

.Ldone:
    movq %r8, %rax
    xorq %rbx, %rbx
    addq %r10, %rax
    adcq $0, %rbx
    addq %r9, %rax
    adcq $0, %rbx
    movq %rax, (%rdi)

    popq %r13
    popq %r12
    popq %rbx
    ret
.size zfactor_fixint_sqr_diag_addlsh1_x86_64, .-zfactor_fixint_sqr_diag_addlsh1_x86_64
