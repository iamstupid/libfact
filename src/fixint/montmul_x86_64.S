	.file	"montmul.ll"
	.text
	.globl	zfactor_montmul1                # -- Begin function zfactor_montmul1
	.p2align	4
	.type	zfactor_montmul1,@function
zfactor_montmul1:                       # @zfactor_montmul1
# %bb.0:                                # %entry
	movq	(%rdx), %rdx
	mulxq	(%rsi), %rsi, %rax
	movq	%r8, %rdx
	movq	(%rcx), %rcx
	imulq	%rsi, %rdx
	mulxq	%rcx, %rdx, %r8
	addq	%rsi, %rdx
	adcq	%rax, %r8
	setb	%al
	movzbl	%al, %eax
	xorl	%edx, %edx
	movq	%r8, %rsi
	subq	%rcx, %rsi
	cmovaeq	%rsi, %r8
	cmpq	$1, %rax
	sbbq	%rdx, %rdx
	cmovaeq	%rsi, %r8
	movq	%r8, (%rdi)
	retq
.Lfunc_end0:
	.size	zfactor_montmul1, .Lfunc_end0-zfactor_montmul1
                                        # -- End function
	.globl	zfactor_montmul2                # -- Begin function zfactor_montmul2
	.p2align	4
	.type	zfactor_montmul2,@function
zfactor_montmul2:                       # @zfactor_montmul2
# %bb.0:                                # %entry
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	movq	%rdx, %rax
	movq	%rdi, -8(%rsp)                  # 8-byte Spill
	movq	(%rsi), %r10
	movq	8(%rsi), %r9
	movq	(%rcx), %rsi
	movq	8(%rcx), %rdi
	movq	(%rdx), %rdx
	mulxq	%r10, %rbx, %r11
	mulxq	%r9, %r15, %r13
	movq	8(%rax), %rcx
	addq	%r11, %r15
	adcq	$0, %r13
	setb	%dl
	movzbl	%dl, %r11d
	movq	%r8, %rdx
	imulq	%rbx, %rdx
	mulxq	%rsi, %rax, %rbp
	mulxq	%rdi, %r12, %r14
	addq	%rbp, %r12
	adcq	$0, %r14
	setb	%dl
	movzbl	%dl, %ebp
	addq	%rbx, %rax
	adcq	%r15, %r12
	adcq	%r13, %r14
	adcq	%r11, %rbp
	setb	%al
	movq	%rcx, %rdx
	mulxq	%r10, %r11, %r15
	movzbl	%al, %eax
	mulxq	%r9, %rbx, %r10
	addq	%r15, %rbx
	adcq	$0, %r10
	setb	%cl
	movzbl	%cl, %r15d
	addq	%r12, %r11
	adcq	%r14, %rbx
	adcq	%rbp, %r10
	adcq	%rax, %r15
	setb	%al
	movzbl	%al, %r9d
	imulq	%r11, %r8
	movq	%r8, %rdx
	mulxq	%rsi, %r14, %rcx
	mulxq	%rdi, %rdx, %rax
	addq	%rcx, %rdx
	adcq	$0, %rax
	setb	%cl
	movzbl	%cl, %ecx
	addq	%r11, %r14
	adcq	%rbx, %rdx
	adcq	%r10, %rax
	adcq	%r15, %rcx
	adcq	$0, %r9
	setb	%r8b
	movzbl	%r8b, %r8d
	movq	%rdx, %r10
	subq	%rsi, %r10
	movq	%rax, %rsi
	sbbq	%rdi, %rsi
	setb	%dil
	xorl	%r11d, %r11d
	cmpq	$1, %rcx
	sbbq	$0, %r9
	sbbq	$0, %r8
	movl	$0, %ecx
	sbbq	%rcx, %rcx
	sbbq	%r11, %r11
	setb	%cl
	testb	%dil, %cl
	cmovneq	%rdx, %r10
	cmovneq	%rax, %rsi
	movq	-8(%rsp), %rax                  # 8-byte Reload
	movq	%r10, (%rax)
	movq	%rsi, 8(%rax)
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
.Lfunc_end1:
	.size	zfactor_montmul2, .Lfunc_end1-zfactor_montmul2
                                        # -- End function
	.globl	zfactor_montmul3                # -- Begin function zfactor_montmul3
	.p2align	4
	.type	zfactor_montmul3,@function
zfactor_montmul3:                       # @zfactor_montmul3
# %bb.0:                                # %entry
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	movq	%rdx, -48(%rsp)                 # 8-byte Spill
	movq	(%rsi), %r9
	movq	%r9, -8(%rsp)                   # 8-byte Spill
	movq	8(%rsi), %r11
	movq	%r11, -40(%rsp)                 # 8-byte Spill
	movq	(%rcx), %rax
	movq	%rax, -16(%rsp)                 # 8-byte Spill
	movq	(%rdx), %rdx
	mulxq	%r9, %rbp, %r10
	movq	16(%rsi), %r9
	movq	%r9, -32(%rsp)                  # 8-byte Spill
	mulxq	%r11, %r13, %r15
	movq	%r8, %rsi
	imulq	%rbp, %rsi
	addq	%r10, %r13
	mulxq	%r9, %r11, %r10
	movq	%rsi, %rdx
	mulxq	%rax, %rbx, %r9
	adcq	%r15, %r11
	movq	8(%rcx), %rax
	movq	%rax, -56(%rsp)                 # 8-byte Spill
	mulxq	%rax, %r15, %r14
	setb	%al
	addq	%r13, %r15
	movq	16(%rcx), %rcx
	movq	%rcx, -24(%rsp)                 # 8-byte Spill
	adcq	$0, %r14
	mulxq	%rcx, %r12, %r13
	addq	%rbp, %rbx
	adcq	%r9, %r15
	adcq	$0, %r14
	addq	%r11, %r12
	movq	-48(%rsp), %rcx                 # 8-byte Reload
	movq	8(%rcx), %rdx
	adcq	$0, %r13
	addq	%r14, %r12
	adcq	$0, %r13
	addb	$255, %al
	movq	-8(%rsp), %r14                  # 8-byte Reload
	mulxq	%r14, %rbp, %rax
	adcq	%r10, %r13
	mulxq	-40(%rsp), %r9, %r10            # 8-byte Folded Reload
	setb	%sil
	addq	%r12, %r9
	adcq	$0, %r10
	addq	%r15, %rbp
	adcq	%rax, %r9
	mulxq	-32(%rsp), %rax, %r15           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%r13, %rax
	adcq	$0, %r15
	addq	%r10, %rax
	adcq	$0, %r15
	movq	%r8, %rdx
	imulq	%rbp, %rdx
	mulxq	-16(%rsp), %r11, %r10           # 8-byte Folded Reload
	movq	-56(%rsp), %rcx                 # 8-byte Reload
	mulxq	%rcx, %r12, %rbx
	addq	%r9, %r12
	adcq	$0, %rbx
	addq	%rbp, %r11
	adcq	%r10, %r12
	adcq	$0, %rbx
	mulxq	-24(%rsp), %r9, %r13            # 8-byte Folded Reload
	addq	%rax, %r9
	adcq	$0, %r13
	addq	%rbx, %r9
	adcq	$0, %r13
	movq	-48(%rsp), %rax                 # 8-byte Reload
	movq	16(%rax), %rdx
	mulxq	%r14, %r14, %r10
	addb	$255, %sil
	mulxq	-40(%rsp), %rbx, %r11           # 8-byte Folded Reload
	adcq	%r15, %r13
	setb	%al
	addq	%r9, %rbx
	adcq	$0, %r11
	addq	%r12, %r14
	mulxq	-32(%rsp), %r15, %rsi           # 8-byte Folded Reload
	adcq	%r10, %rbx
	adcq	$0, %r11
	addq	%r13, %r15
	adcq	$0, %rsi
	addq	%r11, %r15
	adcq	$0, %rsi
	movl	%eax, %edx
	addb	$255, %dl
	movq	%rsi, %r12
	adcq	$0, %r12
	imulq	%r14, %r8
	movq	%r8, %rdx
	movq	-16(%rsp), %rbp                 # 8-byte Reload
	mulxq	%rbp, %r10, %r9
	mulxq	%rcx, %r11, %r13
	addq	%rbx, %r11
	adcq	$0, %r13
	addq	%r14, %r10
	adcq	%r9, %r11
	adcq	$0, %r13
	movq	-24(%rsp), %rcx                 # 8-byte Reload
	mulxq	%rcx, %rdx, %r8
	addq	%r15, %rdx
	adcq	$0, %r8
	addq	%r13, %rdx
	adcq	$0, %r8
	addb	$255, %al
	movq	%r8, %rax
	adcq	%rsi, %rax
	cmpq	%rsi, %r12
	setae	%sil
	cmpq	%r8, %rax
	setae	%r8b
	xorl	%r9d, %r9d
	movq	%r11, %r10
	subq	%rbp, %r10
	movq	%rdx, %rbx
	sbbq	-56(%rsp), %rbx                 # 8-byte Folded Reload
	movq	%rax, %r14
	sbbq	%rcx, %r14
	sbbq	%r9, %r9
	setne	%cl
	andb	%sil, %r8b
	testb	%cl, %r8b
	cmovneq	%r11, %r10
	cmovneq	%rdx, %rbx
	cmovneq	%rax, %r14
	movq	%r10, (%rdi)
	movq	%rbx, 8(%rdi)
	movq	%r14, 16(%rdi)
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
.Lfunc_end2:
	.size	zfactor_montmul3, .Lfunc_end2-zfactor_montmul3
                                        # -- End function
	.globl	zfactor_montmul4                # -- Begin function zfactor_montmul4
	.p2align	4
	.type	zfactor_montmul4,@function
zfactor_montmul4:                       # @zfactor_montmul4
# %bb.0:                                # %entry
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	movq	%rdx, -40(%rsp)                 # 8-byte Spill
	movq	(%rsi), %r9
	movq	%r9, -64(%rsp)                  # 8-byte Spill
	movq	8(%rsi), %r10
	movq	%r10, -56(%rsp)                 # 8-byte Spill
	movq	16(%rsi), %rbx
	movq	%rbx, -48(%rsp)                 # 8-byte Spill
	movq	24(%rsi), %r15
	movq	%r15, -32(%rsp)                 # 8-byte Spill
	movq	(%rdx), %rdx
	mulxq	%r9, %rax, %rsi
	mulxq	%r10, %r10, %r9
	mulxq	%rbx, %r11, %r14
	mulxq	%r15, %rbx, %rdx
	movq	%rdx, -88(%rsp)                 # 8-byte Spill
	movq	%r8, -16(%rsp)                  # 8-byte Spill
	movq	%r8, %rdx
	imulq	%rax, %rdx
	movq	(%rcx), %r8
	movq	%r8, -80(%rsp)                  # 8-byte Spill
	mulxq	%r8, %rbp, %r15
	addq	%rsi, %r10
	movq	8(%rcx), %rsi
	mulxq	%rsi, %r13, %r12
	movq	%rsi, -8(%rsp)                  # 8-byte Spill
	adcq	%r9, %r11
	adcq	%r14, %rbx
	setb	%r9b
	addq	%r10, %r13
	adcq	$0, %r12
	addq	%rax, %rbp
	adcq	%r15, %r13
	movq	16(%rcx), %rax
	movq	%rax, -72(%rsp)                 # 8-byte Spill
	mulxq	%rax, %rax, %r14
	adcq	$0, %r12
	addq	%r11, %rax
	movq	24(%rcx), %rcx
	movq	%rcx, -24(%rsp)                 # 8-byte Spill
	adcq	$0, %r14
	mulxq	%rcx, %r15, %r11
	addq	%r12, %rax
	adcq	$0, %r14
	addq	%rbx, %r15
	adcq	$0, %r11
	addq	%r14, %r15
	adcq	$0, %r11
	addb	$255, %r9b
	adcq	-88(%rsp), %r11                 # 8-byte Folded Reload
	movq	-40(%rsp), %rbp                 # 8-byte Reload
	movq	8(%rbp), %rdx
	mulxq	-64(%rsp), %rbx, %rcx           # 8-byte Folded Reload
	mulxq	-56(%rsp), %r9, %r12            # 8-byte Folded Reload
	setb	-88(%rsp)                       # 1-byte Folded Spill
	addq	%rax, %r9
	adcq	$0, %r12
	addq	%r13, %rbx
	adcq	%rcx, %r9
	mulxq	-48(%rsp), %r14, %rax           # 8-byte Folded Reload
	adcq	$0, %r12
	addq	%r15, %r14
	adcq	$0, %rax
	addq	%r12, %r14
	movq	-32(%rsp), %r8                  # 8-byte Reload
	mulxq	%r8, %r15, %r13
	adcq	$0, %rax
	addq	%r11, %r15
	adcq	$0, %r13
	addq	%rax, %r15
	adcq	$0, %r13
	movq	-16(%rsp), %rdx                 # 8-byte Reload
	imulq	%rbx, %rdx
	movq	-80(%rsp), %r10                 # 8-byte Reload
	mulxq	%r10, %rcx, %rax
	mulxq	%rsi, %r12, %r11
	addq	%r9, %r12
	adcq	$0, %r11
	addq	%rbx, %rcx
	adcq	%rax, %r12
	adcq	$0, %r11
	mulxq	-72(%rsp), %rax, %rcx           # 8-byte Folded Reload
	addq	%r14, %rax
	adcq	$0, %rcx
	addq	%r11, %rax
	adcq	$0, %rcx
	movq	-24(%rsp), %rsi                 # 8-byte Reload
	mulxq	%rsi, %r9, %r11
	addq	%r15, %r9
	adcq	$0, %r11
	addq	%rcx, %r9
	adcq	$0, %r11
	addb	$255, -88(%rsp)                 # 1-byte Folded Spill
	adcq	%r13, %r11
	movq	16(%rbp), %rdx
	mulxq	-64(%rsp), %rbx, %r13           # 8-byte Folded Reload
	mulxq	-56(%rsp), %r15, %rbp           # 8-byte Folded Reload
	setb	-88(%rsp)                       # 1-byte Folded Spill
	addq	%rax, %r15
	adcq	$0, %rbp
	mulxq	-48(%rsp), %r14, %rax           # 8-byte Folded Reload
	addq	%r12, %rbx
	adcq	%r13, %r15
	adcq	$0, %rbp
	addq	%r9, %r14
	adcq	$0, %rax
	addq	%rbp, %r14
	mulxq	%r8, %r9, %r12
	adcq	$0, %rax
	addq	%r11, %r9
	adcq	$0, %r12
	addq	%rax, %r9
	adcq	$0, %r12
	movq	-16(%rsp), %r8                  # 8-byte Reload
	movq	%r8, %rdx
	imulq	%rbx, %rdx
	mulxq	%r10, %r11, %rax
	movq	-8(%rsp), %r10                  # 8-byte Reload
	mulxq	%r10, %r13, %rcx
	addq	%r15, %r13
	adcq	$0, %rcx
	addq	%rbx, %r11
	adcq	%rax, %r13
	adcq	$0, %rcx
	mulxq	-72(%rsp), %rbp, %rax           # 8-byte Folded Reload
	addq	%r14, %rbp
	adcq	$0, %rax
	addq	%rcx, %rbp
	adcq	$0, %rax
	mulxq	%rsi, %r11, %rbx
	addq	%r9, %r11
	adcq	$0, %rbx
	addq	%rax, %r11
	adcq	$0, %rbx
	addb	$255, -88(%rsp)                 # 1-byte Folded Spill
	movq	-40(%rsp), %rax                 # 8-byte Reload
	movq	24(%rax), %rdx
	mulxq	-64(%rsp), %r15, %rcx           # 8-byte Folded Reload
	mulxq	-56(%rsp), %r14, %rsi           # 8-byte Folded Reload
	adcq	%r12, %rbx
	setb	%al
	addq	%rbp, %r14
	adcq	$0, %rsi
	addq	%r13, %r15
	adcq	%rcx, %r14
	mulxq	-48(%rsp), %r12, %rcx           # 8-byte Folded Reload
	adcq	$0, %rsi
	addq	%r11, %r12
	adcq	$0, %rcx
	addq	%rsi, %r12
	mulxq	-32(%rsp), %r13, %r11           # 8-byte Folded Reload
	adcq	$0, %rcx
	addq	%rbx, %r13
	adcq	$0, %r11
	addq	%rcx, %r13
	adcq	$0, %r11
	movl	%eax, %ecx
	addb	$255, %cl
	movq	%r11, %rbp
	adcq	$0, %rbp
	imulq	%r15, %r8
	movq	%r8, %rdx
	mulxq	-80(%rsp), %rsi, %rcx           # 8-byte Folded Reload
	mulxq	%r10, %rbx, %r9
	addq	%r14, %rbx
	adcq	$0, %r9
	addq	%r15, %rsi
	adcq	%rcx, %rbx
	adcq	$0, %r9
	movq	-72(%rsp), %r15                 # 8-byte Reload
	mulxq	%r15, %r14, %rsi
	addq	%r12, %r14
	adcq	$0, %rsi
	addq	%r9, %r14
	adcq	$0, %rsi
	movq	-24(%rsp), %r12                 # 8-byte Reload
	mulxq	%r12, %rcx, %rdx
	addq	%r13, %rcx
	adcq	$0, %rdx
	addq	%rsi, %rcx
	adcq	$0, %rdx
	addb	$255, %al
	movq	%rdx, %rax
	adcq	%r11, %rax
	cmpq	%r11, %rbp
	setae	%sil
	cmpq	%rdx, %rax
	setae	%dl
	movq	%rbx, %r8
	subq	-80(%rsp), %r8                  # 8-byte Folded Reload
	movq	%r14, %r9
	sbbq	%r10, %r9
	movq	%rcx, %r11
	sbbq	%r15, %r11
	movq	%rax, %r15
	sbbq	%r12, %r15
	setb	%r10b
	andb	%sil, %dl
	testb	%r10b, %dl
	cmovneq	%rbx, %r8
	cmovneq	%r14, %r9
	cmovneq	%rcx, %r11
	cmovneq	%rax, %r15
	movq	%r8, (%rdi)
	movq	%r9, 8(%rdi)
	movq	%r11, 16(%rdi)
	movq	%r15, 24(%rdi)
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
.Lfunc_end3:
	.size	zfactor_montmul4, .Lfunc_end3-zfactor_montmul4
                                        # -- End function
	.globl	zfactor_montmul5                # -- Begin function zfactor_montmul5
	.p2align	4
	.type	zfactor_montmul5,@function
zfactor_montmul5:                       # @zfactor_montmul5
# %bb.0:                                # %entry
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	movq	%r8, -64(%rsp)                  # 8-byte Spill
	movq	%rdx, -96(%rsp)                 # 8-byte Spill
	movq	(%rsi), %rax
	movq	%rax, -8(%rsp)                  # 8-byte Spill
	movq	8(%rsi), %rbx
	movq	%rbx, -16(%rsp)                 # 8-byte Spill
	movq	16(%rsi), %r11
	movq	%r11, -40(%rsp)                 # 8-byte Spill
	movq	24(%rsi), %r10
	movq	%r10, -88(%rsp)                 # 8-byte Spill
	movq	(%rdx), %rdx
	mulxq	%rax, %r14, %rax
	mulxq	%rbx, %r13, %r9
	movq	32(%rsi), %rbx
	movq	%rbx, -80(%rsp)                 # 8-byte Spill
	movq	%r8, %rsi
	imulq	%r14, %rsi
	addq	%rax, %r13
	mulxq	%r11, %r12, %rax
	mulxq	%r10, %r11, %r15
	mulxq	%rbx, %r8, %r10
	adcq	%r9, %r12
	adcq	%rax, %r11
	movq	(%rcx), %rax
	movq	%rax, -48(%rsp)                 # 8-byte Spill
	movq	%rsi, %rdx
	mulxq	%rax, %rbx, %rbp
	adcq	%r15, %r8
	movq	8(%rcx), %rax
	movq	%rax, -56(%rsp)                 # 8-byte Spill
	mulxq	%rax, %r15, %r9
	setb	-97(%rsp)                       # 1-byte Folded Spill
	addq	%r13, %r15
	adcq	$0, %r9
	movq	16(%rcx), %rax
	movq	%rax, -72(%rsp)                 # 8-byte Spill
	mulxq	%rax, %rax, %r13
	addq	%r14, %rbx
	adcq	%rbp, %r15
	adcq	$0, %r9
	addq	%r12, %rax
	adcq	$0, %r13
	addq	%r9, %rax
	movq	24(%rcx), %r9
	movq	%r9, -24(%rsp)                  # 8-byte Spill
	mulxq	%r9, %r14, %r9
	adcq	$0, %r13
	addq	%r11, %r14
	adcq	$0, %r9
	addq	%r13, %r14
	movq	32(%rcx), %rcx
	movq	%rcx, -32(%rsp)                 # 8-byte Spill
	mulxq	%rcx, %r11, %r12
	adcq	$0, %r9
	addq	%r8, %r11
	adcq	$0, %r12
	addq	%r9, %r11
	adcq	$0, %r12
	movq	-96(%rsp), %rcx                 # 8-byte Reload
	movq	8(%rcx), %rdx
	movq	-8(%rsp), %rsi                  # 8-byte Reload
	mulxq	%rsi, %r13, %rcx
	mulxq	-16(%rsp), %rbp, %r8            # 8-byte Folded Reload
	addb	$255, -97(%rsp)                 # 1-byte Folded Spill
	adcq	%r10, %r12
	setb	-97(%rsp)                       # 1-byte Folded Spill
	addq	%rax, %rbp
	adcq	$0, %r8
	addq	%r15, %r13
	adcq	%rcx, %rbp
	mulxq	-40(%rsp), %rcx, %rax           # 8-byte Folded Reload
	adcq	$0, %r8
	addq	%r14, %rcx
	adcq	$0, %rax
	addq	%r8, %rcx
	mulxq	-88(%rsp), %r8, %r10            # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r11, %r8
	adcq	$0, %r10
	addq	%rax, %r8
	mulxq	-80(%rsp), %r9, %r14            # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%r12, %r9
	adcq	$0, %r14
	addq	%r10, %r9
	adcq	$0, %r14
	movq	-64(%rsp), %rdx                 # 8-byte Reload
	imulq	%r13, %rdx
	mulxq	-48(%rsp), %r10, %rax           # 8-byte Folded Reload
	mulxq	-56(%rsp), %r12, %r11           # 8-byte Folded Reload
	addq	%rbp, %r12
	adcq	$0, %r11
	addq	%r13, %r10
	mulxq	-72(%rsp), %rbx, %r10           # 8-byte Folded Reload
	adcq	%rax, %r12
	adcq	$0, %r11
	addq	%rcx, %rbx
	adcq	$0, %r10
	addq	%r11, %rbx
	adcq	$0, %r10
	mulxq	-24(%rsp), %r11, %rcx           # 8-byte Folded Reload
	addq	%r8, %r11
	adcq	$0, %rcx
	addq	%r10, %r11
	adcq	$0, %rcx
	mulxq	-32(%rsp), %rax, %r13           # 8-byte Folded Reload
	addq	%r9, %rax
	adcq	$0, %r13
	addq	%rcx, %rax
	adcq	$0, %r13
	movq	-96(%rsp), %rcx                 # 8-byte Reload
	movq	16(%rcx), %rdx
	mulxq	%rsi, %r15, %r9
	addb	$255, -97(%rsp)                 # 1-byte Folded Spill
	adcq	%r14, %r13
	movq	-16(%rsp), %rbp                 # 8-byte Reload
	mulxq	%rbp, %r8, %r10
	setb	-97(%rsp)                       # 1-byte Folded Spill
	addq	%rbx, %r8
	adcq	$0, %r10
	addq	%r12, %r15
	mulxq	-40(%rsp), %rcx, %rbx           # 8-byte Folded Reload
	adcq	%r9, %r8
	adcq	$0, %r10
	addq	%r11, %rcx
	mulxq	-88(%rsp), %r9, %r11            # 8-byte Folded Reload
	adcq	$0, %rbx
	addq	%r10, %rcx
	adcq	$0, %rbx
	addq	%rax, %r9
	adcq	$0, %r11
	addq	%rbx, %r9
	mulxq	-80(%rsp), %r12, %r14           # 8-byte Folded Reload
	adcq	$0, %r11
	addq	%r13, %r12
	adcq	$0, %r14
	addq	%r11, %r12
	adcq	$0, %r14
	movq	-64(%rsp), %rdx                 # 8-byte Reload
	imulq	%r15, %rdx
	mulxq	-48(%rsp), %r10, %rax           # 8-byte Folded Reload
	movq	-56(%rsp), %rsi                 # 8-byte Reload
	mulxq	%rsi, %r13, %r11
	addq	%r8, %r13
	adcq	$0, %r11
	addq	%r15, %r10
	adcq	%rax, %r13
	adcq	$0, %r11
	mulxq	-72(%rsp), %r8, %r10            # 8-byte Folded Reload
	addq	%rcx, %r8
	adcq	$0, %r10
	addq	%r11, %r8
	adcq	$0, %r10
	mulxq	-24(%rsp), %rax, %rbx           # 8-byte Folded Reload
	addq	%r9, %rax
	adcq	$0, %rbx
	addq	%r10, %rax
	mulxq	-32(%rsp), %r11, %rcx           # 8-byte Folded Reload
	adcq	$0, %rbx
	addq	%r12, %r11
	adcq	$0, %rcx
	addq	%rbx, %r11
	adcq	$0, %rcx
	movq	-96(%rsp), %rdx                 # 8-byte Reload
	movq	24(%rdx), %rdx
	mulxq	-8(%rsp), %r12, %r10            # 8-byte Folded Reload
	mulxq	%rbp, %r9, %rbx
	addb	$255, -97(%rsp)                 # 1-byte Folded Spill
	adcq	%r14, %rcx
	setb	-97(%rsp)                       # 1-byte Folded Spill
	addq	%r8, %r9
	adcq	$0, %rbx
	addq	%r13, %r12
	adcq	%r10, %r9
	mulxq	-40(%rsp), %r8, %r10            # 8-byte Folded Reload
	adcq	$0, %rbx
	addq	%rax, %r8
	adcq	$0, %r10
	addq	%rbx, %r8
	mulxq	-88(%rsp), %r13, %rax           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%r11, %r13
	adcq	$0, %rax
	addq	%r10, %r13
	mulxq	-80(%rsp), %rbp, %r15           # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%rcx, %rbp
	adcq	$0, %r15
	addq	%rax, %rbp
	adcq	$0, %r15
	movq	-64(%rsp), %rdx                 # 8-byte Reload
	imulq	%r12, %rdx
	movq	-48(%rsp), %rbx                 # 8-byte Reload
	mulxq	%rbx, %rcx, %rax
	mulxq	%rsi, %r14, %r10
	addq	%r9, %r14
	adcq	$0, %r10
	addq	%r12, %rcx
	adcq	%rax, %r14
	adcq	$0, %r10
	mulxq	-72(%rsp), %r9, %rax            # 8-byte Folded Reload
	addq	%r8, %r9
	adcq	$0, %rax
	addq	%r10, %r9
	adcq	$0, %rax
	mulxq	-24(%rsp), %rcx, %r8            # 8-byte Folded Reload
	addq	%r13, %rcx
	adcq	$0, %r8
	addq	%rax, %rcx
	adcq	$0, %r8
	mulxq	-32(%rsp), %rax, %r12           # 8-byte Folded Reload
	addq	%rbp, %rax
	adcq	$0, %r12
	addq	%r8, %rax
	adcq	$0, %r12
	addb	$255, -97(%rsp)                 # 1-byte Folded Spill
	adcq	%r15, %r12
	movq	-96(%rsp), %rdx                 # 8-byte Reload
	movq	32(%rdx), %rdx
	mulxq	-8(%rsp), %r15, %rsi            # 8-byte Folded Reload
	mulxq	-16(%rsp), %r13, %r10           # 8-byte Folded Reload
	setb	%r8b
	movb	%r8b, -96(%rsp)                 # 1-byte Spill
	addq	%r9, %r13
	adcq	$0, %r10
	mulxq	-40(%rsp), %r9, %r11            # 8-byte Folded Reload
	addq	%r14, %r15
	adcq	%rsi, %r13
	adcq	$0, %r10
	addq	%rcx, %r9
	adcq	$0, %r11
	addq	%r10, %r9
	mulxq	-88(%rsp), %rbp, %rcx           # 8-byte Folded Reload
	adcq	$0, %r11
	addq	%rax, %rbp
	adcq	$0, %rcx
	addq	%r11, %rbp
	mulxq	-80(%rsp), %rax, %r14           # 8-byte Folded Reload
	adcq	$0, %rcx
	addq	%r12, %rax
	adcq	$0, %r14
	addq	%rcx, %rax
	adcq	$0, %r14
	movl	%r8d, %ecx
	addb	$255, %cl
	movq	%r14, %rcx
	adcq	$0, %rcx
	movq	-64(%rsp), %rdx                 # 8-byte Reload
	imulq	%r15, %rdx
	mulxq	%rbx, %r11, %r10
	mulxq	-56(%rsp), %r8, %rbx            # 8-byte Folded Reload
	addq	%r13, %r8
	adcq	$0, %rbx
	addq	%r15, %r11
	adcq	%r10, %r8
	movq	%r8, -88(%rsp)                  # 8-byte Spill
	adcq	$0, %rbx
	movq	-72(%rsp), %r13                 # 8-byte Reload
	mulxq	%r13, %r15, %r10
	addq	%r9, %r15
	adcq	$0, %r10
	addq	%rbx, %r15
	adcq	$0, %r10
	movq	-24(%rsp), %r11                 # 8-byte Reload
	mulxq	%r11, %r12, %r9
	addq	%rbp, %r12
	adcq	$0, %r9
	addq	%r10, %r12
	adcq	$0, %r9
	movq	-32(%rsp), %rbp                 # 8-byte Reload
	mulxq	%rbp, %rdx, %r10
	addq	%rax, %rdx
	adcq	$0, %r10
	addq	%r9, %rdx
	adcq	$0, %r10
	addb	$255, -96(%rsp)                 # 1-byte Folded Spill
	movq	%r10, %rsi
	adcq	%r14, %rsi
	cmpq	%r14, %rcx
	setae	-80(%rsp)                       # 1-byte Folded Spill
	cmpq	%r10, %rsi
	setae	%bl
	xorl	%r14d, %r14d
	movq	%r8, %rcx
	subq	-48(%rsp), %rcx                 # 8-byte Folded Reload
	movq	%r15, %r8
	sbbq	-56(%rsp), %r8                  # 8-byte Folded Reload
	movq	%r12, %r9
	sbbq	%r13, %r9
	movq	%rdx, %r10
	sbbq	%r11, %r10
	movq	%rsi, %r13
	sbbq	%rbp, %r13
	movl	$0, %ebp
	sbbq	%rbp, %rbp
	movl	$0, %eax
	sbbq	%rax, %rax
	sbbq	%r14, %r14
	orq	%rbp, %r14
	orq	%rax, %r14
	setne	%al
	andb	-80(%rsp), %bl                  # 1-byte Folded Reload
	testb	%al, %bl
	cmovneq	-88(%rsp), %rcx                 # 8-byte Folded Reload
	cmovneq	%r15, %r8
	cmovneq	%r12, %r9
	cmovneq	%rdx, %r10
	cmovneq	%rsi, %r13
	movq	%rcx, (%rdi)
	movq	%r8, 8(%rdi)
	movq	%r9, 16(%rdi)
	movq	%r10, 24(%rdi)
	movq	%r13, 32(%rdi)
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
.Lfunc_end4:
	.size	zfactor_montmul5, .Lfunc_end4-zfactor_montmul5
                                        # -- End function
	.globl	zfactor_montmul6                # -- Begin function zfactor_montmul6
	.p2align	4
	.type	zfactor_montmul6,@function
zfactor_montmul6:                       # @zfactor_montmul6
# %bb.0:                                # %entry
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	movq	%r8, -88(%rsp)                  # 8-byte Spill
	movq	%rdx, -48(%rsp)                 # 8-byte Spill
	movq	(%rsi), %rax
	movq	%rax, -56(%rsp)                 # 8-byte Spill
	movq	8(%rsi), %r9
	movq	%r9, -40(%rsp)                  # 8-byte Spill
	movq	16(%rsi), %rbx
	movq	%rbx, -32(%rsp)                 # 8-byte Spill
	movq	(%rdx), %rdx
	mulxq	%rax, %r14, %rax
	movq	%r14, -112(%rsp)                # 8-byte Spill
	mulxq	%r9, %r10, %r11
	movq	%r8, %r9
	imulq	%r14, %r9
	addq	%rax, %r10
	mulxq	%rbx, %r15, %r12
	adcq	%r11, %r15
	movq	24(%rsi), %r11
	movq	%r11, -24(%rsp)                 # 8-byte Spill
	movq	32(%rsi), %rax
	movq	%rax, -104(%rsp)                # 8-byte Spill
	movq	40(%rsi), %r8
	movq	%r8, -96(%rsp)                  # 8-byte Spill
	mulxq	%r11, %rbx, %rsi
	mulxq	%rax, %r14, %r13
	mulxq	%r8, %rax, %rdx
	movq	%rdx, -120(%rsp)                # 8-byte Spill
	adcq	%r12, %rbx
	adcq	%rsi, %r14
	movq	(%rcx), %rsi
	movq	%rsi, -80(%rsp)                 # 8-byte Spill
	movq	%r9, %rdx
	mulxq	%rsi, %r8, %r12
	adcq	%r13, %rax
	movq	8(%rcx), %rsi
	movq	%rsi, -64(%rsp)                 # 8-byte Spill
	mulxq	%rsi, %rbp, %r13
	setb	-121(%rsp)                      # 1-byte Folded Spill
	addq	%r10, %rbp
	adcq	$0, %r13
	movq	16(%rcx), %rsi
	movq	%rsi, -72(%rsp)                 # 8-byte Spill
	mulxq	%rsi, %r10, %rsi
	addq	-112(%rsp), %r8                 # 8-byte Folded Reload
	adcq	%r12, %rbp
	adcq	$0, %r13
	addq	%r15, %r10
	adcq	$0, %rsi
	addq	%r13, %r10
	movq	24(%rcx), %r8
	movq	%r8, -16(%rsp)                  # 8-byte Spill
	mulxq	%r8, %r11, %r8
	adcq	$0, %rsi
	addq	%rbx, %r11
	adcq	$0, %r8
	addq	%rsi, %r11
	movq	32(%rcx), %rsi
	movq	%rsi, -112(%rsp)                # 8-byte Spill
	mulxq	%rsi, %rbx, %rsi
	adcq	$0, %r8
	addq	%r14, %rbx
	movq	40(%rcx), %rcx
	movq	%rcx, -8(%rsp)                  # 8-byte Spill
	mulxq	%rcx, %r13, %r15
	adcq	$0, %rsi
	addq	%r8, %rbx
	adcq	$0, %rsi
	addq	%rax, %r13
	adcq	$0, %r15
	addq	%rsi, %r13
	adcq	$0, %r15
	addb	$255, -121(%rsp)                # 1-byte Folded Spill
	movq	-48(%rsp), %rax                 # 8-byte Reload
	movq	8(%rax), %rdx
	mulxq	-56(%rsp), %r12, %rax           # 8-byte Folded Reload
	adcq	-120(%rsp), %r15                # 8-byte Folded Reload
	mulxq	-40(%rsp), %r8, %r9             # 8-byte Folded Reload
	setb	-120(%rsp)                      # 1-byte Folded Spill
	addq	%r10, %r8
	adcq	$0, %r9
	addq	%rbp, %r12
	adcq	%rax, %r8
	mulxq	-32(%rsp), %rsi, %r10           # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%r11, %rsi
	adcq	$0, %r10
	mulxq	-24(%rsp), %rcx, %r11           # 8-byte Folded Reload
	addq	%r9, %rsi
	adcq	$0, %r10
	addq	%rbx, %rcx
	mulxq	-104(%rsp), %rax, %r9           # 8-byte Folded Reload
	adcq	$0, %r11
	addq	%r10, %rcx
	adcq	$0, %r11
	addq	%r13, %rax
	adcq	$0, %r9
	addq	%r11, %rax
	mulxq	-96(%rsp), %r10, %r13           # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%r15, %r10
	adcq	$0, %r13
	addq	%r9, %r10
	adcq	$0, %r13
	movq	-88(%rsp), %rdx                 # 8-byte Reload
	imulq	%r12, %rdx
	mulxq	-80(%rsp), %r11, %r9            # 8-byte Folded Reload
	mulxq	-64(%rsp), %rbp, %rbx           # 8-byte Folded Reload
	addq	%r8, %rbp
	adcq	$0, %rbx
	addq	%r12, %r11
	adcq	%r9, %rbp
	adcq	$0, %rbx
	mulxq	-72(%rsp), %r14, %r8            # 8-byte Folded Reload
	addq	%rsi, %r14
	adcq	$0, %r8
	addq	%rbx, %r14
	mulxq	-16(%rsp), %r9, %rsi            # 8-byte Folded Reload
	adcq	$0, %r8
	addq	%rcx, %r9
	adcq	$0, %rsi
	mulxq	-112(%rsp), %r11, %rcx          # 8-byte Folded Reload
	addq	%r8, %r9
	adcq	$0, %rsi
	addq	%rax, %r11
	adcq	$0, %rcx
	addq	%rsi, %r11
	adcq	$0, %rcx
	mulxq	-8(%rsp), %rbx, %r15            # 8-byte Folded Reload
	addq	%r10, %rbx
	adcq	$0, %r15
	addq	%rcx, %rbx
	adcq	$0, %r15
	addb	$255, -120(%rsp)                # 1-byte Folded Spill
	movq	-48(%rsp), %rax                 # 8-byte Reload
	movq	16(%rax), %rdx
	mulxq	-56(%rsp), %r12, %rax           # 8-byte Folded Reload
	adcq	%r13, %r15
	mulxq	-40(%rsp), %r8, %r10            # 8-byte Folded Reload
	setb	-120(%rsp)                      # 1-byte Folded Spill
	addq	%r14, %r8
	adcq	$0, %r10
	addq	%rbp, %r12
	adcq	%rax, %r8
	mulxq	-32(%rsp), %rsi, %r14           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%r9, %rsi
	adcq	$0, %r14
	mulxq	-24(%rsp), %rcx, %r9            # 8-byte Folded Reload
	addq	%r10, %rsi
	adcq	$0, %r14
	addq	%r11, %rcx
	mulxq	-104(%rsp), %rax, %r11          # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%r14, %rcx
	adcq	$0, %r9
	addq	%rbx, %rax
	adcq	$0, %r11
	addq	%r9, %rax
	mulxq	-96(%rsp), %r10, %r13           # 8-byte Folded Reload
	adcq	$0, %r11
	addq	%r15, %r10
	adcq	$0, %r13
	addq	%r11, %r10
	adcq	$0, %r13
	movq	-88(%rsp), %rdx                 # 8-byte Reload
	imulq	%r12, %rdx
	mulxq	-80(%rsp), %r11, %r9            # 8-byte Folded Reload
	mulxq	-64(%rsp), %rbp, %rbx           # 8-byte Folded Reload
	addq	%r8, %rbp
	adcq	$0, %rbx
	addq	%r12, %r11
	adcq	%r9, %rbp
	adcq	$0, %rbx
	mulxq	-72(%rsp), %r14, %r8            # 8-byte Folded Reload
	addq	%rsi, %r14
	adcq	$0, %r8
	addq	%rbx, %r14
	mulxq	-16(%rsp), %r11, %rsi           # 8-byte Folded Reload
	adcq	$0, %r8
	addq	%rcx, %r11
	adcq	$0, %rsi
	mulxq	-112(%rsp), %rbx, %rcx          # 8-byte Folded Reload
	addq	%r8, %r11
	adcq	$0, %rsi
	addq	%rax, %rbx
	adcq	$0, %rcx
	addq	%rsi, %rbx
	adcq	$0, %rcx
	mulxq	-8(%rsp), %r9, %r15             # 8-byte Folded Reload
	addq	%r10, %r9
	adcq	$0, %r15
	addq	%rcx, %r9
	adcq	$0, %r15
	addb	$255, -120(%rsp)                # 1-byte Folded Spill
	movq	-48(%rsp), %rax                 # 8-byte Reload
	movq	24(%rax), %rdx
	mulxq	-56(%rsp), %r12, %rax           # 8-byte Folded Reload
	adcq	%r13, %r15
	mulxq	-40(%rsp), %r8, %r10            # 8-byte Folded Reload
	setb	-120(%rsp)                      # 1-byte Folded Spill
	addq	%r14, %r8
	adcq	$0, %r10
	addq	%rbp, %r12
	adcq	%rax, %r8
	mulxq	-32(%rsp), %rsi, %r14           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%r11, %rsi
	adcq	$0, %r14
	mulxq	-24(%rsp), %rcx, %r11           # 8-byte Folded Reload
	addq	%r10, %rsi
	adcq	$0, %r14
	addq	%rbx, %rcx
	mulxq	-104(%rsp), %rax, %rbx          # 8-byte Folded Reload
	adcq	$0, %r11
	addq	%r14, %rcx
	adcq	$0, %r11
	addq	%r9, %rax
	adcq	$0, %rbx
	addq	%r11, %rax
	mulxq	-96(%rsp), %r10, %r13           # 8-byte Folded Reload
	adcq	$0, %rbx
	addq	%r15, %r10
	adcq	$0, %r13
	addq	%rbx, %r10
	adcq	$0, %r13
	movq	-88(%rsp), %rdx                 # 8-byte Reload
	imulq	%r12, %rdx
	mulxq	-80(%rsp), %r11, %r9            # 8-byte Folded Reload
	mulxq	-64(%rsp), %rbp, %rbx           # 8-byte Folded Reload
	addq	%r8, %rbp
	adcq	$0, %rbx
	addq	%r12, %r11
	adcq	%r9, %rbp
	adcq	$0, %rbx
	mulxq	-72(%rsp), %r9, %r8             # 8-byte Folded Reload
	addq	%rsi, %r9
	adcq	$0, %r8
	addq	%rbx, %r9
	mulxq	-16(%rsp), %rsi, %rbx           # 8-byte Folded Reload
	adcq	$0, %r8
	addq	%rcx, %rsi
	adcq	$0, %rbx
	mulxq	-112(%rsp), %r11, %rcx          # 8-byte Folded Reload
	addq	%r8, %rsi
	adcq	$0, %rbx
	addq	%rax, %r11
	adcq	$0, %rcx
	addq	%rbx, %r11
	adcq	$0, %rcx
	movq	-8(%rsp), %rax                  # 8-byte Reload
	mulxq	%rax, %rbx, %r15
	addq	%r10, %rbx
	adcq	$0, %r15
	addq	%rcx, %rbx
	adcq	$0, %r15
	addb	$255, -120(%rsp)                # 1-byte Folded Spill
	movq	-48(%rsp), %rcx                 # 8-byte Reload
	movq	32(%rcx), %rdx
	mulxq	-56(%rsp), %r12, %rcx           # 8-byte Folded Reload
	adcq	%r13, %r15
	mulxq	-40(%rsp), %r8, %r10            # 8-byte Folded Reload
	setb	-120(%rsp)                      # 1-byte Folded Spill
	addq	%r9, %r8
	adcq	$0, %r10
	addq	%rbp, %r12
	adcq	%rcx, %r8
	mulxq	-32(%rsp), %rcx, %r14           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%rsi, %rcx
	adcq	$0, %r14
	mulxq	-24(%rsp), %r9, %rsi            # 8-byte Folded Reload
	addq	%r10, %rcx
	adcq	$0, %r14
	addq	%r11, %r9
	mulxq	-104(%rsp), %r10, %r11          # 8-byte Folded Reload
	adcq	$0, %rsi
	addq	%r14, %r9
	adcq	$0, %rsi
	addq	%rbx, %r10
	adcq	$0, %r11
	addq	%rsi, %r10
	mulxq	-96(%rsp), %rsi, %r13           # 8-byte Folded Reload
	adcq	$0, %r11
	addq	%r15, %rsi
	adcq	$0, %r13
	addq	%r11, %rsi
	adcq	$0, %r13
	movq	-88(%rsp), %rdx                 # 8-byte Reload
	imulq	%r12, %rdx
	mulxq	-80(%rsp), %rbx, %r11           # 8-byte Folded Reload
	mulxq	-64(%rsp), %rbp, %r15           # 8-byte Folded Reload
	addq	%r8, %rbp
	adcq	$0, %r15
	addq	%r12, %rbx
	adcq	%r11, %rbp
	mulxq	-72(%rsp), %r14, %r8            # 8-byte Folded Reload
	adcq	$0, %r15
	addq	%rcx, %r14
	adcq	$0, %r8
	mulxq	-16(%rsp), %rcx, %rbx           # 8-byte Folded Reload
	addq	%r15, %r14
	adcq	$0, %r8
	addq	%r9, %rcx
	adcq	$0, %rbx
	addq	%r8, %rcx
	adcq	$0, %rbx
	mulxq	-112(%rsp), %r11, %r8           # 8-byte Folded Reload
	addq	%r10, %r11
	adcq	$0, %r8
	addq	%rbx, %r11
	adcq	$0, %r8
	mulxq	%rax, %rbx, %r9
	addq	%rsi, %rbx
	adcq	$0, %r9
	addq	%r8, %rbx
	adcq	$0, %r9
	movq	-48(%rsp), %rax                 # 8-byte Reload
	movq	40(%rax), %rdx
	mulxq	-56(%rsp), %r12, %rsi           # 8-byte Folded Reload
	addb	$255, -120(%rsp)                # 1-byte Folded Spill
	adcq	%r13, %r9
	mulxq	-40(%rsp), %r10, %rax           # 8-byte Folded Reload
	setb	%r8b
	addq	%r14, %r10
	adcq	$0, %rax
	addq	%rbp, %r12
	mulxq	-32(%rsp), %r13, %r14           # 8-byte Folded Reload
	adcq	%rsi, %r10
	adcq	$0, %rax
	addq	%rcx, %r13
	mulxq	-24(%rsp), %rbp, %rcx           # 8-byte Folded Reload
	adcq	$0, %r14
	addq	%rax, %r13
	adcq	$0, %r14
	addq	%r11, %rbp
	adcq	$0, %rcx
	addq	%r14, %rbp
	mulxq	-104(%rsp), %rax, %rsi          # 8-byte Folded Reload
	adcq	$0, %rcx
	addq	%rbx, %rax
	adcq	$0, %rsi
	addq	%rcx, %rax
	mulxq	-96(%rsp), %rcx, %r15           # 8-byte Folded Reload
	adcq	$0, %rsi
	addq	%r9, %rcx
	adcq	$0, %r15
	addq	%rsi, %rcx
	adcq	$0, %r15
	movl	%r8d, %edx
	addb	$255, %dl
	movq	%r15, %rsi
	adcq	$0, %rsi
	movq	-88(%rsp), %rdx                 # 8-byte Reload
	imulq	%r12, %rdx
	mulxq	-80(%rsp), %rbx, %r11           # 8-byte Folded Reload
	movq	%rdx, %r9
	mulxq	-64(%rsp), %rdx, %r14           # 8-byte Folded Reload
	movq	%rdi, -88(%rsp)                 # 8-byte Spill
	addq	%r10, %rdx
	adcq	$0, %r14
	addq	%r12, %rbx
	adcq	%r11, %rdx
	movq	%rdx, -96(%rsp)                 # 8-byte Spill
	adcq	$0, %r14
	movq	%r9, %rdx
	mulxq	-72(%rsp), %r12, %r11           # 8-byte Folded Reload
	addq	%r13, %r12
	adcq	$0, %r11
	addq	%r14, %r12
	adcq	$0, %r11
	movq	-16(%rsp), %r13                 # 8-byte Reload
	mulxq	%r13, %r10, %rbx
	addq	%rbp, %r10
	adcq	$0, %rbx
	addq	%r11, %r10
	adcq	$0, %rbx
	mulxq	-112(%rsp), %r11, %r14          # 8-byte Folded Reload
	addq	%rax, %r11
	adcq	$0, %r14
	addq	%rbx, %r11
	adcq	$0, %r14
	movq	-8(%rsp), %rax                  # 8-byte Reload
	mulxq	%rax, %rdi, %rdx
	addq	%rcx, %rdi
	adcq	$0, %rdx
	addq	%r14, %rdi
	adcq	$0, %rdx
	addb	$255, %r8b
	movq	%rdx, %r9
	adcq	%r15, %r9
	cmpq	%r15, %rsi
	setae	-104(%rsp)                      # 1-byte Folded Spill
	cmpq	%rdx, %r9
	setae	%r14b
	xorl	%r15d, %r15d
	movq	-96(%rsp), %rbp                 # 8-byte Reload
	movq	%rbp, %rdx
	subq	-80(%rsp), %rdx                 # 8-byte Folded Reload
	movq	%r12, %rsi
	sbbq	-64(%rsp), %rsi                 # 8-byte Folded Reload
	movq	%r10, %r8
	sbbq	-72(%rsp), %r8                  # 8-byte Folded Reload
	movq	%r11, %rbx
	sbbq	%r13, %rbx
	movq	%rdi, %r13
	sbbq	-112(%rsp), %r13                # 8-byte Folded Reload
	movq	%r9, %rcx
	sbbq	%rax, %rcx
	movl	$0, %eax
	sbbq	%rax, %rax
	sbbq	%r15, %r15
	orq	%rax, %r15
	setne	%al
	andb	-104(%rsp), %r14b               # 1-byte Folded Reload
	testb	%al, %r14b
	cmovneq	%rbp, %rdx
	cmovneq	%r12, %rsi
	cmovneq	%r10, %r8
	cmovneq	%r11, %rbx
	cmovneq	%rdi, %r13
	cmovneq	%r9, %rcx
	movq	-88(%rsp), %rax                 # 8-byte Reload
	movq	%rdx, (%rax)
	movq	%rsi, 8(%rax)
	movq	%r8, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	%r13, 32(%rax)
	movq	%rcx, 40(%rax)
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
.Lfunc_end5:
	.size	zfactor_montmul6, .Lfunc_end5-zfactor_montmul6
                                        # -- End function
	.globl	zfactor_montmul7                # -- Begin function zfactor_montmul7
	.p2align	4
	.type	zfactor_montmul7,@function
zfactor_montmul7:                       # @zfactor_montmul7
# %bb.0:                                # %entry
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$16, %rsp
	movq	%r8, -56(%rsp)                  # 8-byte Spill
	movq	%rdx, -32(%rsp)                 # 8-byte Spill
	movq	(%rsi), %rbx
	movq	%rbx, -40(%rsp)                 # 8-byte Spill
	movq	8(%rsi), %rax
	movq	%rax, -80(%rsp)                 # 8-byte Spill
	movq	16(%rsi), %r11
	movq	%r11, -24(%rsp)                 # 8-byte Spill
	movq	24(%rsi), %r10
	movq	%r10, -16(%rsp)                 # 8-byte Spill
	movq	32(%rsi), %r9
	movq	%r9, -8(%rsp)                   # 8-byte Spill
	movq	(%rdx), %rdx
	mulxq	%rbx, %r15, %rbx
	movq	%rbx, -88(%rsp)                 # 8-byte Spill
	movq	%r15, -104(%rsp)                # 8-byte Spill
	mulxq	%rax, %r14, %rbp
	movq	40(%rsi), %rbx
	movq	%rbx, -72(%rsp)                 # 8-byte Spill
	movq	48(%rsi), %rax
	movq	%rax, -64(%rsp)                 # 8-byte Spill
	movq	%r8, %rsi
	imulq	%r15, %rsi
	mulxq	%r11, %r13, %r8
	movq	%r8, -96(%rsp)                  # 8-byte Spill
	mulxq	%r10, %r15, %r8
	mulxq	%r9, %r12, %r9
	movq	%r9, -112(%rsp)                 # 8-byte Spill
	mulxq	%rbx, %r11, %r10
	mulxq	%rax, %r9, %rax
	movq	%rax, -128(%rsp)                # 8-byte Spill
	addq	-88(%rsp), %r14                 # 8-byte Folded Reload
	adcq	%rbp, %r13
	movq	(%rcx), %rax
	movq	%rax, -88(%rsp)                 # 8-byte Spill
	movq	%rsi, %rdx
	mulxq	%rax, %rbx, %rbp
	adcq	-96(%rsp), %r15                 # 8-byte Folded Reload
	adcq	%r8, %r12
	movq	8(%rcx), %rax
	movq	%rax, -96(%rsp)                 # 8-byte Spill
	mulxq	%rax, %rax, %r8
	adcq	-112(%rsp), %r11                # 8-byte Folded Reload
	movq	%r11, -120(%rsp)                # 8-byte Spill
	adcq	%r10, %r9
	movq	%r9, (%rsp)                     # 8-byte Spill
	setb	%r11b
	addq	%r14, %rax
	adcq	$0, %r8
	addq	-104(%rsp), %rbx                # 8-byte Folded Reload
	adcq	%rbp, %rax
	movq	16(%rcx), %r9
	movq	%r9, -48(%rsp)                  # 8-byte Spill
	mulxq	%r9, %r14, %r10
	adcq	$0, %r8
	addq	%r13, %r14
	adcq	$0, %r10
	movq	24(%rcx), %r9
	movq	%r9, 8(%rsp)                    # 8-byte Spill
	mulxq	%r9, %rbx, %r13
	addq	%r8, %r14
	adcq	$0, %r10
	addq	%r15, %rbx
	adcq	$0, %r13
	addq	%r10, %rbx
	movq	32(%rcx), %r8
	movq	%r8, -104(%rsp)                 # 8-byte Spill
	mulxq	%r8, %r15, %r8
	adcq	$0, %r13
	addq	%r12, %r15
	adcq	$0, %r8
	addq	%r13, %r15
	movq	40(%rcx), %r9
	movq	%r9, -112(%rsp)                 # 8-byte Spill
	mulxq	%r9, %r12, %r10
	adcq	$0, %r8
	addq	-120(%rsp), %r12                # 8-byte Folded Reload
	movq	48(%rcx), %rcx
	movq	%rcx, -120(%rsp)                # 8-byte Spill
	adcq	$0, %r10
	mulxq	%rcx, %rcx, %r9
	addq	%r8, %r12
	adcq	$0, %r10
	addq	(%rsp), %rcx                    # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%r10, %rcx
	adcq	$0, %r9
	addb	$255, %r11b
	adcq	-128(%rsp), %r9                 # 8-byte Folded Reload
	movq	-32(%rsp), %rdx                 # 8-byte Reload
	movq	8(%rdx), %rdx
	mulxq	-40(%rsp), %r8, %rsi            # 8-byte Folded Reload
	mulxq	-80(%rsp), %rbp, %r10           # 8-byte Folded Reload
	setb	-128(%rsp)                      # 1-byte Folded Spill
	addq	%r14, %rbp
	adcq	$0, %r10
	addq	%rax, %r8
	adcq	%rsi, %rbp
	mulxq	-24(%rsp), %r14, %rax           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%rbx, %r14
	adcq	$0, %rax
	addq	%r10, %r14
	mulxq	-16(%rsp), %r10, %rsi           # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r15, %r10
	adcq	$0, %rsi
	mulxq	-8(%rsp), %r11, %r15            # 8-byte Folded Reload
	addq	%rax, %r10
	adcq	$0, %rsi
	addq	%r12, %r11
	mulxq	-72(%rsp), %rbx, %rax           # 8-byte Folded Reload
	adcq	$0, %r15
	addq	%rsi, %r11
	adcq	$0, %r15
	addq	%rcx, %rbx
	adcq	$0, %rax
	addq	%r15, %rbx
	mulxq	-64(%rsp), %r15, %rsi           # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r9, %r15
	adcq	$0, %rsi
	addq	%rax, %r15
	adcq	$0, %rsi
	movq	-56(%rsp), %rdx                 # 8-byte Reload
	imulq	%r8, %rdx
	mulxq	-88(%rsp), %r9, %rcx            # 8-byte Folded Reload
	mulxq	-96(%rsp), %rax, %r13           # 8-byte Folded Reload
	addq	%rbp, %rax
	adcq	$0, %r13
	addq	%r8, %r9
	adcq	%rcx, %rax
	mulxq	-48(%rsp), %r12, %rcx           # 8-byte Folded Reload
	adcq	$0, %r13
	addq	%r14, %r12
	adcq	$0, %rcx
	mulxq	8(%rsp), %r14, %r8              # 8-byte Folded Reload
	addq	%r13, %r12
	adcq	$0, %rcx
	addq	%r10, %r14
	adcq	$0, %r8
	addq	%rcx, %r14
	adcq	$0, %r8
	mulxq	-104(%rsp), %r10, %rcx          # 8-byte Folded Reload
	addq	%r11, %r10
	adcq	$0, %rcx
	addq	%r8, %r10
	adcq	$0, %rcx
	mulxq	-112(%rsp), %r13, %r9           # 8-byte Folded Reload
	addq	%rbx, %r13
	adcq	$0, %r9
	addq	%rcx, %r13
	adcq	$0, %r9
	mulxq	-120(%rsp), %rcx, %r8           # 8-byte Folded Reload
	addq	%r15, %rcx
	adcq	$0, %r8
	addq	%r9, %rcx
	adcq	$0, %r8
	addb	$255, -128(%rsp)                # 1-byte Folded Spill
	adcq	%rsi, %r8
	movq	-32(%rsp), %rdx                 # 8-byte Reload
	movq	16(%rdx), %rdx
	mulxq	-40(%rsp), %r9, %rsi            # 8-byte Folded Reload
	mulxq	-80(%rsp), %rbx, %r15           # 8-byte Folded Reload
	setb	-128(%rsp)                      # 1-byte Folded Spill
	addq	%r12, %rbx
	adcq	$0, %r15
	mulxq	-24(%rsp), %r11, %r12           # 8-byte Folded Reload
	addq	%rax, %r9
	adcq	%rsi, %rbx
	adcq	$0, %r15
	addq	%r14, %r11
	adcq	$0, %r12
	addq	%r15, %r11
	mulxq	-16(%rsp), %r15, %rax           # 8-byte Folded Reload
	adcq	$0, %r12
	addq	%r10, %r15
	adcq	$0, %rax
	addq	%r12, %r15
	mulxq	-8(%rsp), %r12, %rsi            # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r13, %r12
	adcq	$0, %rsi
	addq	%rax, %r12
	mulxq	-72(%rsp), %r14, %r13           # 8-byte Folded Reload
	adcq	$0, %rsi
	addq	%rcx, %r14
	adcq	$0, %r13
	addq	%rsi, %r14
	mulxq	-64(%rsp), %r10, %rax           # 8-byte Folded Reload
	adcq	$0, %r13
	addq	%r8, %r10
	adcq	$0, %rax
	addq	%r13, %r10
	adcq	$0, %rax
	movq	-56(%rsp), %rdx                 # 8-byte Reload
	imulq	%r9, %rdx
	mulxq	-88(%rsp), %r8, %rsi            # 8-byte Folded Reload
	mulxq	-96(%rsp), %rcx, %rbp           # 8-byte Folded Reload
	addq	%rbx, %rcx
	adcq	$0, %rbp
	addq	%r9, %r8
	adcq	%rsi, %rcx
	adcq	$0, %rbp
	mulxq	-48(%rsp), %r13, %rsi           # 8-byte Folded Reload
	addq	%r11, %r13
	adcq	$0, %rsi
	addq	%rbp, %r13
	adcq	$0, %rsi
	mulxq	8(%rsp), %r11, %r8              # 8-byte Folded Reload
	addq	%r15, %r11
	adcq	$0, %r8
	addq	%rsi, %r11
	adcq	$0, %r8
	mulxq	-104(%rsp), %rsi, %r9           # 8-byte Folded Reload
	addq	%r12, %rsi
	adcq	$0, %r9
	addq	%r8, %rsi
	mulxq	-112(%rsp), %rbx, %r12          # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%r14, %rbx
	adcq	$0, %r12
	mulxq	-120(%rsp), %r15, %r8           # 8-byte Folded Reload
	addq	%r9, %rbx
	adcq	$0, %r12
	addq	%r10, %r15
	adcq	$0, %r8
	addq	%r12, %r15
	adcq	$0, %r8
	addb	$255, -128(%rsp)                # 1-byte Folded Spill
	adcq	%rax, %r8
	movq	-32(%rsp), %rax                 # 8-byte Reload
	movq	24(%rax), %rdx
	mulxq	-40(%rsp), %r9, %rax            # 8-byte Folded Reload
	mulxq	-80(%rsp), %rbp, %r10           # 8-byte Folded Reload
	setb	-128(%rsp)                      # 1-byte Folded Spill
	addq	%r13, %rbp
	adcq	$0, %r10
	addq	%rcx, %r9
	adcq	%rax, %rbp
	mulxq	-24(%rsp), %r14, %rax           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%r11, %r14
	adcq	$0, %rax
	addq	%r10, %r14
	mulxq	-16(%rsp), %r10, %rcx           # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%rsi, %r10
	adcq	$0, %rcx
	mulxq	-8(%rsp), %r11, %rsi            # 8-byte Folded Reload
	addq	%rax, %r10
	adcq	$0, %rcx
	addq	%rbx, %r11
	mulxq	-72(%rsp), %rbx, %rax           # 8-byte Folded Reload
	adcq	$0, %rsi
	addq	%rcx, %r11
	adcq	$0, %rsi
	addq	%r15, %rbx
	adcq	$0, %rax
	addq	%rsi, %rbx
	mulxq	-64(%rsp), %r15, %rcx           # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r8, %r15
	adcq	$0, %rcx
	addq	%rax, %r15
	adcq	$0, %rcx
	movq	-56(%rsp), %rdx                 # 8-byte Reload
	imulq	%r9, %rdx
	mulxq	-88(%rsp), %r8, %rsi            # 8-byte Folded Reload
	mulxq	-96(%rsp), %rax, %r12           # 8-byte Folded Reload
	addq	%rbp, %rax
	adcq	$0, %r12
	addq	%r9, %r8
	adcq	%rsi, %rax
	mulxq	-48(%rsp), %rbp, %rsi           # 8-byte Folded Reload
	adcq	$0, %r12
	addq	%r14, %rbp
	adcq	$0, %rsi
	mulxq	8(%rsp), %r14, %r8              # 8-byte Folded Reload
	addq	%r12, %rbp
	adcq	$0, %rsi
	addq	%r10, %r14
	adcq	$0, %r8
	addq	%rsi, %r14
	adcq	$0, %r8
	mulxq	-104(%rsp), %rsi, %r9           # 8-byte Folded Reload
	addq	%r11, %rsi
	adcq	$0, %r9
	addq	%r8, %rsi
	adcq	$0, %r9
	mulxq	-112(%rsp), %r10, %r11          # 8-byte Folded Reload
	addq	%rbx, %r10
	adcq	$0, %r11
	addq	%r9, %r10
	adcq	$0, %r11
	mulxq	-120(%rsp), %r12, %r8           # 8-byte Folded Reload
	addq	%r15, %r12
	adcq	$0, %r8
	addq	%r11, %r12
	adcq	$0, %r8
	addb	$255, -128(%rsp)                # 1-byte Folded Spill
	adcq	%rcx, %r8
	movq	-32(%rsp), %rcx                 # 8-byte Reload
	movq	32(%rcx), %rdx
	mulxq	-40(%rsp), %r13, %rcx           # 8-byte Folded Reload
	mulxq	-80(%rsp), %rbx, %r15           # 8-byte Folded Reload
	setb	(%rsp)                          # 1-byte Folded Spill
	addq	%rbp, %rbx
	adcq	$0, %r15
	mulxq	-24(%rsp), %r11, %r9            # 8-byte Folded Reload
	addq	%rax, %r13
	adcq	%rcx, %rbx
	adcq	$0, %r15
	addq	%r14, %r11
	adcq	$0, %r9
	addq	%r15, %r11
	mulxq	-16(%rsp), %rbp, %rax           # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%rsi, %rbp
	adcq	$0, %rax
	addq	%r9, %rbp
	mulxq	-8(%rsp), %rsi, %rcx            # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r10, %rsi
	adcq	$0, %rcx
	addq	%rax, %rsi
	mulxq	-72(%rsp), %r14, %r9            # 8-byte Folded Reload
	adcq	$0, %rcx
	addq	%r12, %r14
	adcq	$0, %r9
	addq	%rcx, %r14
	mulxq	-64(%rsp), %r10, %rax           # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%r8, %r10
	adcq	$0, %rax
	addq	%r9, %r10
	adcq	$0, %rax
	movq	-56(%rsp), %rdx                 # 8-byte Reload
	imulq	%r13, %rdx
	mulxq	-88(%rsp), %r9, %r8             # 8-byte Folded Reload
	mulxq	-96(%rsp), %rcx, %r15           # 8-byte Folded Reload
	movq	%rdi, -128(%rsp)                # 8-byte Spill
	addq	%rbx, %rcx
	adcq	$0, %r15
	addq	%r13, %r9
	adcq	%r8, %rcx
	adcq	$0, %r15
	mulxq	-48(%rsp), %rdi, %r8            # 8-byte Folded Reload
	addq	%r11, %rdi
	adcq	$0, %r8
	addq	%r15, %rdi
	adcq	$0, %r8
	movq	8(%rsp), %r13                   # 8-byte Reload
	mulxq	%r13, %rbx, %r9
	addq	%rbp, %rbx
	adcq	$0, %r9
	addq	%r8, %rbx
	adcq	$0, %r9
	mulxq	-104(%rsp), %r12, %r11          # 8-byte Folded Reload
	addq	%rsi, %r12
	adcq	$0, %r11
	addq	%r9, %r12
	mulxq	-112(%rsp), %r9, %rsi           # 8-byte Folded Reload
	adcq	$0, %r11
	addq	%r14, %r9
	adcq	$0, %rsi
	mulxq	-120(%rsp), %r14, %r8           # 8-byte Folded Reload
	addq	%r11, %r9
	adcq	$0, %rsi
	addq	%r10, %r14
	adcq	$0, %r8
	addq	%rsi, %r14
	adcq	$0, %r8
	addb	$255, (%rsp)                    # 1-byte Folded Spill
	adcq	%rax, %r8
	movq	-32(%rsp), %rax                 # 8-byte Reload
	movq	40(%rax), %rdx
	mulxq	-40(%rsp), %rsi, %rax           # 8-byte Folded Reload
	mulxq	-80(%rsp), %rbp, %r10           # 8-byte Folded Reload
	setb	(%rsp)                          # 1-byte Folded Spill
	addq	%rdi, %rbp
	adcq	$0, %r10
	addq	%rcx, %rsi
	adcq	%rax, %rbp
	mulxq	-24(%rsp), %r11, %rax           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%rbx, %r11
	adcq	$0, %rax
	addq	%r10, %r11
	mulxq	-16(%rsp), %rbx, %rcx           # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r12, %rbx
	adcq	$0, %rcx
	mulxq	-8(%rsp), %r10, %rdi            # 8-byte Folded Reload
	addq	%rax, %rbx
	adcq	$0, %rcx
	addq	%r9, %r10
	mulxq	-72(%rsp), %r9, %rax            # 8-byte Folded Reload
	adcq	$0, %rdi
	addq	%rcx, %r10
	adcq	$0, %rdi
	addq	%r14, %r9
	adcq	$0, %rax
	addq	%rdi, %r9
	mulxq	-64(%rsp), %r14, %r15           # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r8, %r14
	adcq	$0, %r15
	addq	%rax, %r14
	adcq	$0, %r15
	movq	-56(%rsp), %rdx                 # 8-byte Reload
	imulq	%rsi, %rdx
	mulxq	-88(%rsp), %rdi, %rax           # 8-byte Folded Reload
	mulxq	-96(%rsp), %rcx, %r8            # 8-byte Folded Reload
	addq	%rbp, %rcx
	adcq	$0, %r8
	addq	%rsi, %rdi
	adcq	%rax, %rcx
	adcq	$0, %r8
	mulxq	-48(%rsp), %rax, %rdi           # 8-byte Folded Reload
	addq	%r11, %rax
	adcq	$0, %rdi
	addq	%r8, %rax
	adcq	$0, %rdi
	mulxq	%r13, %rsi, %r8
	addq	%rbx, %rsi
	adcq	$0, %r8
	addq	%rdi, %rsi
	mulxq	-104(%rsp), %r12, %r11          # 8-byte Folded Reload
	adcq	$0, %r8
	addq	%r10, %r12
	adcq	$0, %r11
	mulxq	-112(%rsp), %rdi, %r10          # 8-byte Folded Reload
	addq	%r8, %r12
	adcq	$0, %r11
	addq	%r9, %rdi
	adcq	$0, %r10
	addq	%r11, %rdi
	adcq	$0, %r10
	mulxq	-120(%rsp), %r9, %r8            # 8-byte Folded Reload
	addq	%r14, %r9
	adcq	$0, %r8
	addq	%r10, %r9
	adcq	$0, %r8
	addb	$255, (%rsp)                    # 1-byte Folded Spill
	movq	-32(%rsp), %rdx                 # 8-byte Reload
	movq	48(%rdx), %rdx
	mulxq	-40(%rsp), %r13, %r10           # 8-byte Folded Reload
	adcq	%r15, %r8
	mulxq	-80(%rsp), %r15, %r14           # 8-byte Folded Reload
	setb	%bpl
	movb	%bpl, -80(%rsp)                 # 1-byte Spill
	addq	%rax, %r15
	adcq	$0, %r14
	addq	%rcx, %r13
	adcq	%r10, %r15
	mulxq	-24(%rsp), %r11, %rax           # 8-byte Folded Reload
	adcq	$0, %r14
	addq	%rsi, %r11
	adcq	$0, %rax
	mulxq	-16(%rsp), %rbx, %rcx           # 8-byte Folded Reload
	addq	%r14, %r11
	adcq	$0, %rax
	addq	%r12, %rbx
	mulxq	-8(%rsp), %r10, %rsi            # 8-byte Folded Reload
	adcq	$0, %rcx
	addq	%rax, %rbx
	adcq	$0, %rcx
	addq	%rdi, %r10
	adcq	$0, %rsi
	addq	%rcx, %r10
	mulxq	-72(%rsp), %rdi, %rax           # 8-byte Folded Reload
	adcq	$0, %rsi
	addq	%r9, %rdi
	adcq	$0, %rax
	addq	%rsi, %rdi
	mulxq	-64(%rsp), %r9, %rcx            # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r8, %r9
	adcq	$0, %rcx
	addq	%rax, %r9
	adcq	$0, %rcx
	addb	$255, %bpl
	movq	%rcx, %rax
	adcq	$0, %rax
	movq	%rax, -64(%rsp)                 # 8-byte Spill
	movq	-56(%rsp), %rdx                 # 8-byte Reload
	imulq	%r13, %rdx
	mulxq	-88(%rsp), %r8, %r14            # 8-byte Folded Reload
	mulxq	-96(%rsp), %rax, %r12           # 8-byte Folded Reload
	addq	%r15, %rax
	adcq	$0, %r12
	addq	%r13, %r8
	mulxq	-48(%rsp), %rbp, %r15           # 8-byte Folded Reload
	adcq	%r14, %rax
	movq	%rax, %r14
	movq	%rax, -72(%rsp)                 # 8-byte Spill
	adcq	$0, %r12
	addq	%r11, %rbp
	adcq	$0, %r15
	addq	%r12, %rbp
	adcq	$0, %r15
	movq	8(%rsp), %r13                   # 8-byte Reload
	mulxq	%r13, %r8, %r12
	addq	%rbx, %r8
	adcq	$0, %r12
	addq	%r15, %r8
	adcq	$0, %r12
	mulxq	-104(%rsp), %r11, %rbx          # 8-byte Folded Reload
	addq	%r10, %r11
	adcq	$0, %rbx
	addq	%r12, %r11
	adcq	$0, %rbx
	mulxq	-112(%rsp), %r10, %r15          # 8-byte Folded Reload
	addq	%rdi, %r10
	adcq	$0, %r15
	addq	%rbx, %r10
	adcq	$0, %r15
	mulxq	-120(%rsp), %rax, %rbx          # 8-byte Folded Reload
	addq	%r9, %rax
	adcq	$0, %rbx
	addq	%r15, %rax
	adcq	$0, %rbx
	addb	$255, -80(%rsp)                 # 1-byte Folded Spill
	movq	%rbx, %rsi
	adcq	%rcx, %rsi
	cmpq	%rcx, -64(%rsp)                 # 8-byte Folded Reload
	setae	-56(%rsp)                       # 1-byte Folded Spill
	cmpq	%rbx, %rsi
	setae	%r15b
	xorl	%r12d, %r12d
	movq	%r14, %rcx
	subq	-88(%rsp), %rcx                 # 8-byte Folded Reload
	movq	%rbp, %r9
	sbbq	-96(%rsp), %r9                  # 8-byte Folded Reload
	movq	%r8, %rbx
	sbbq	-48(%rsp), %rbx                 # 8-byte Folded Reload
	movq	%r11, %r14
	sbbq	%r13, %r14
	movq	%r10, %r13
	sbbq	-104(%rsp), %r13                # 8-byte Folded Reload
	movq	%rax, %rdx
	sbbq	-112(%rsp), %rdx                # 8-byte Folded Reload
	movq	%rsi, %rdi
	sbbq	-120(%rsp), %rdi                # 8-byte Folded Reload
	sbbq	%r12, %r12
	setne	%r12b
	andb	-56(%rsp), %r15b                # 1-byte Folded Reload
	testb	%r12b, %r15b
	cmovneq	-72(%rsp), %rcx                 # 8-byte Folded Reload
	cmovneq	%rbp, %r9
	cmovneq	%r8, %rbx
	cmovneq	%r11, %r14
	cmovneq	%r10, %r13
	cmovneq	%rax, %rdx
	cmovneq	%rsi, %rdi
	movq	-128(%rsp), %rax                # 8-byte Reload
	movq	%rcx, (%rax)
	movq	%r9, 8(%rax)
	movq	%rbx, 16(%rax)
	movq	%r14, 24(%rax)
	movq	%r13, 32(%rax)
	movq	%rdx, 40(%rax)
	movq	%rdi, 48(%rax)
	addq	$16, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
.Lfunc_end6:
	.size	zfactor_montmul7, .Lfunc_end6-zfactor_montmul7
                                        # -- End function
	.globl	zfactor_montmul8                # -- Begin function zfactor_montmul8
	.p2align	4
	.type	zfactor_montmul8,@function
zfactor_montmul8:                       # @zfactor_montmul8
# %bb.0:                                # %entry
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$32, %rsp
	movq	%r8, -104(%rsp)                 # 8-byte Spill
	movq	%rdx, -120(%rsp)                # 8-byte Spill
	movq	(%rsi), %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	8(%rsi), %r9
	movq	%r9, 16(%rsp)                   # 8-byte Spill
	movq	16(%rsi), %r10
	movq	%r10, 24(%rsp)                  # 8-byte Spill
	movq	24(%rsi), %r11
	movq	%r11, -56(%rsp)                 # 8-byte Spill
	movq	(%rdx), %rdx
	mulxq	%rax, %rbx, %rax
	movq	%rbx, -72(%rsp)                 # 8-byte Spill
	mulxq	%r9, %r14, %r12
	movq	%r8, %r9
	imulq	%rbx, %r9
	addq	%rax, %r14
	movq	32(%rsi), %rax
	movq	%rax, -48(%rsp)                 # 8-byte Spill
	movq	40(%rsi), %r8
	movq	%r8, -40(%rsp)                  # 8-byte Spill
	movq	48(%rsi), %rbp
	movq	%rbp, -32(%rsp)                 # 8-byte Spill
	movq	56(%rsi), %r15
	movq	%r15, -112(%rsp)                # 8-byte Spill
	mulxq	%r10, %r13, %rsi
	mulxq	%r11, %r10, %r11
	movq	%r11, -64(%rsp)                 # 8-byte Spill
	mulxq	%rax, %r11, %rax
	movq	%rax, -80(%rsp)                 # 8-byte Spill
	mulxq	%r8, %rax, %r8
	movq	%r8, -88(%rsp)                  # 8-byte Spill
	mulxq	%rbp, %rbx, %r8
	mulxq	%r15, %r15, %rdx
	movq	%rdx, -128(%rsp)                # 8-byte Spill
	adcq	%r12, %r13
	adcq	%rsi, %r10
	movq	(%rcx), %rsi
	movq	%rsi, -8(%rsp)                  # 8-byte Spill
	movq	%r9, %rdx
	mulxq	%rsi, %r12, %rsi
	adcq	-64(%rsp), %r11                 # 8-byte Folded Reload
	adcq	-80(%rsp), %rax                 # 8-byte Folded Reload
	movq	%rax, -96(%rsp)                 # 8-byte Spill
	movq	8(%rcx), %rax
	movq	%rax, -64(%rsp)                 # 8-byte Spill
	mulxq	%rax, %rbp, %rax
	adcq	-88(%rsp), %rbx                 # 8-byte Folded Reload
	movq	%rbx, -24(%rsp)                 # 8-byte Spill
	adcq	%r8, %r15
	movq	%r15, (%rsp)                    # 8-byte Spill
	setb	%bl
	addq	%r14, %rbp
	adcq	$0, %rax
	addq	-72(%rsp), %r12                 # 8-byte Folded Reload
	adcq	%rsi, %rbp
	movq	16(%rcx), %rsi
	movq	%rsi, -72(%rsp)                 # 8-byte Spill
	mulxq	%rsi, %r14, %rsi
	adcq	$0, %rax
	addq	%r13, %r14
	adcq	$0, %rsi
	movq	24(%rcx), %r8
	movq	%r8, -16(%rsp)                  # 8-byte Spill
	mulxq	%r8, %r8, %r12
	addq	%rax, %r14
	adcq	$0, %rsi
	addq	%r10, %r8
	adcq	$0, %r12
	addq	%rsi, %r8
	movq	32(%rcx), %rax
	movq	%rax, -80(%rsp)                 # 8-byte Spill
	mulxq	%rax, %r10, %rax
	adcq	$0, %r12
	addq	%r11, %r10
	adcq	$0, %rax
	addq	%r12, %r10
	movq	40(%rcx), %rsi
	movq	%rsi, -88(%rsp)                 # 8-byte Spill
	mulxq	%rsi, %r11, %rsi
	adcq	$0, %rax
	addq	-96(%rsp), %r11                 # 8-byte Folded Reload
	adcq	$0, %rsi
	movq	48(%rcx), %r15
	movq	%r15, -96(%rsp)                 # 8-byte Spill
	mulxq	%r15, %r15, %r12
	addq	%rax, %r11
	adcq	$0, %rsi
	addq	-24(%rsp), %r15                 # 8-byte Folded Reload
	adcq	$0, %r12
	addq	%rsi, %r15
	movq	56(%rcx), %rax
	movq	%rax, -24(%rsp)                 # 8-byte Spill
	mulxq	%rax, %r13, %rcx
	adcq	$0, %r12
	addq	(%rsp), %r13                    # 8-byte Folded Reload
	adcq	$0, %rcx
	addq	%r12, %r13
	adcq	$0, %rcx
	movq	-120(%rsp), %rax                # 8-byte Reload
	movq	8(%rax), %rdx
	mulxq	8(%rsp), %rax, %r9              # 8-byte Folded Reload
	addb	$255, %bl
	adcq	-128(%rsp), %rcx                # 8-byte Folded Reload
	mulxq	16(%rsp), %rbx, %r12            # 8-byte Folded Reload
	setb	-128(%rsp)                      # 1-byte Folded Spill
	addq	%r14, %rbx
	adcq	$0, %r12
	addq	%rbp, %rax
	mulxq	24(%rsp), %rsi, %r14            # 8-byte Folded Reload
	adcq	%r9, %rbx
	adcq	$0, %r12
	addq	%r8, %rsi
	mulxq	-56(%rsp), %r8, %r9             # 8-byte Folded Reload
	adcq	$0, %r14
	addq	%r12, %rsi
	adcq	$0, %r14
	addq	%r10, %r8
	adcq	$0, %r9
	addq	%r14, %r8
	mulxq	-48(%rsp), %r14, %r10           # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%r11, %r14
	adcq	$0, %r10
	addq	%r9, %r14
	mulxq	-40(%rsp), %r11, %r9            # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%r15, %r11
	adcq	$0, %r9
	addq	%r10, %r11
	mulxq	-32(%rsp), %r15, %r12           # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%r13, %r15
	adcq	$0, %r12
	addq	%r9, %r15
	mulxq	-112(%rsp), %r10, %r13          # 8-byte Folded Reload
	adcq	$0, %r12
	addq	%rcx, %r10
	adcq	$0, %r13
	addq	%r12, %r10
	adcq	$0, %r13
	movq	-104(%rsp), %rdx                # 8-byte Reload
	imulq	%rax, %rdx
	mulxq	-8(%rsp), %r12, %r9             # 8-byte Folded Reload
	mulxq	-64(%rsp), %rcx, %rbp           # 8-byte Folded Reload
	addq	%rbx, %rcx
	adcq	$0, %rbp
	addq	%rax, %r12
	mulxq	-72(%rsp), %rbx, %rax           # 8-byte Folded Reload
	adcq	%r9, %rcx
	adcq	$0, %rbp
	addq	%rsi, %rbx
	adcq	$0, %rax
	addq	%rbp, %rbx
	adcq	$0, %rax
	mulxq	-16(%rsp), %r12, %rsi           # 8-byte Folded Reload
	addq	%r8, %r12
	adcq	$0, %rsi
	addq	%rax, %r12
	adcq	$0, %rsi
	mulxq	-80(%rsp), %r9, %rax            # 8-byte Folded Reload
	addq	%r14, %r9
	adcq	$0, %rax
	addq	%rsi, %r9
	adcq	$0, %rax
	mulxq	-88(%rsp), %rbp, %rsi           # 8-byte Folded Reload
	addq	%r11, %rbp
	adcq	$0, %rsi
	addq	%rax, %rbp
	adcq	$0, %rsi
	mulxq	-96(%rsp), %r8, %r11            # 8-byte Folded Reload
	addq	%r15, %r8
	adcq	$0, %r11
	addq	%rsi, %r8
	mulxq	-24(%rsp), %r14, %rax           # 8-byte Folded Reload
	adcq	$0, %r11
	addq	%r10, %r14
	adcq	$0, %rax
	addq	%r11, %r14
	adcq	$0, %rax
	movq	-120(%rsp), %rdx                # 8-byte Reload
	movq	16(%rdx), %rdx
	mulxq	8(%rsp), %r11, %r10             # 8-byte Folded Reload
	mulxq	16(%rsp), %rsi, %r15            # 8-byte Folded Reload
	addb	$255, -128(%rsp)                # 1-byte Folded Spill
	adcq	%r13, %rax
	setb	-128(%rsp)                      # 1-byte Folded Spill
	addq	%rbx, %rsi
	adcq	$0, %r15
	addq	%rcx, %r11
	movq	%r11, (%rsp)                    # 8-byte Spill
	adcq	%r10, %rsi
	mulxq	24(%rsp), %r11, %rcx            # 8-byte Folded Reload
	adcq	$0, %r15
	addq	%r12, %r11
	adcq	$0, %rcx
	addq	%r15, %r11
	mulxq	-56(%rsp), %rbx, %r15           # 8-byte Folded Reload
	adcq	$0, %rcx
	addq	%r9, %rbx
	adcq	$0, %r15
	addq	%rcx, %rbx
	mulxq	-48(%rsp), %r10, %r9            # 8-byte Folded Reload
	adcq	$0, %r15
	addq	%rbp, %r10
	adcq	$0, %r9
	mulxq	-40(%rsp), %rcx, %r12           # 8-byte Folded Reload
	addq	%r15, %r10
	adcq	$0, %r9
	addq	%r8, %rcx
	mulxq	-32(%rsp), %r8, %r15            # 8-byte Folded Reload
	adcq	$0, %r12
	addq	%r9, %rcx
	adcq	$0, %r12
	addq	%r14, %r8
	adcq	$0, %r15
	addq	%r12, %r8
	mulxq	-112(%rsp), %r14, %r13          # 8-byte Folded Reload
	adcq	$0, %r15
	addq	%rax, %r14
	adcq	$0, %r13
	addq	%r15, %r14
	adcq	$0, %r13
	movq	-104(%rsp), %rdx                # 8-byte Reload
	movq	(%rsp), %rbp                    # 8-byte Reload
	imulq	%rbp, %rdx
	mulxq	-8(%rsp), %r15, %r9             # 8-byte Folded Reload
	mulxq	-64(%rsp), %rax, %r12           # 8-byte Folded Reload
	addq	%rsi, %rax
	adcq	$0, %r12
	addq	%rbp, %r15
	adcq	%r9, %rax
	adcq	$0, %r12
	mulxq	-72(%rsp), %rsi, %r9            # 8-byte Folded Reload
	addq	%r11, %rsi
	adcq	$0, %r9
	addq	%r12, %rsi
	adcq	$0, %r9
	mulxq	-16(%rsp), %r15, %r11           # 8-byte Folded Reload
	addq	%rbx, %r15
	adcq	$0, %r11
	addq	%r9, %r15
	mulxq	-80(%rsp), %rbx, %r9            # 8-byte Folded Reload
	adcq	$0, %r11
	addq	%r10, %rbx
	adcq	$0, %r9
	mulxq	-88(%rsp), %r10, %rbp           # 8-byte Folded Reload
	addq	%r11, %rbx
	adcq	$0, %r9
	addq	%rcx, %r10
	adcq	$0, %rbp
	addq	%r9, %r10
	adcq	$0, %rbp
	mulxq	-96(%rsp), %r12, %r11           # 8-byte Folded Reload
	addq	%r8, %r12
	adcq	$0, %r11
	addq	%rbp, %r12
	adcq	$0, %r11
	mulxq	-24(%rsp), %r9, %rcx            # 8-byte Folded Reload
	addq	%r14, %r9
	adcq	$0, %rcx
	addq	%r11, %r9
	adcq	$0, %rcx
	movq	-120(%rsp), %rdx                # 8-byte Reload
	movq	24(%rdx), %rdx
	mulxq	8(%rsp), %rbp, %r8              # 8-byte Folded Reload
	addb	$255, -128(%rsp)                # 1-byte Folded Spill
	adcq	%r13, %rcx
	mulxq	16(%rsp), %r11, %r14            # 8-byte Folded Reload
	setb	-128(%rsp)                      # 1-byte Folded Spill
	addq	%rsi, %r11
	adcq	$0, %r14
	addq	%rax, %rbp
	mulxq	24(%rsp), %rsi, %rax            # 8-byte Folded Reload
	adcq	%r8, %r11
	adcq	$0, %r14
	addq	%r15, %rsi
	mulxq	-56(%rsp), %r8, %r15            # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r14, %rsi
	adcq	$0, %rax
	addq	%rbx, %r8
	adcq	$0, %r15
	addq	%rax, %r8
	mulxq	-48(%rsp), %rbx, %rax           # 8-byte Folded Reload
	adcq	$0, %r15
	addq	%r10, %rbx
	adcq	$0, %rax
	addq	%r15, %rbx
	mulxq	-40(%rsp), %r14, %r10           # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r12, %r14
	adcq	$0, %r10
	addq	%rax, %r14
	mulxq	-32(%rsp), %r15, %rax           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%r9, %r15
	adcq	$0, %rax
	addq	%r10, %r15
	mulxq	-112(%rsp), %r10, %r13          # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%rcx, %r10
	adcq	$0, %r13
	addq	%rax, %r10
	adcq	$0, %r13
	movq	-104(%rsp), %rdx                # 8-byte Reload
	imulq	%rbp, %rdx
	mulxq	-8(%rsp), %r9, %rcx             # 8-byte Folded Reload
	mulxq	-64(%rsp), %rax, %r12           # 8-byte Folded Reload
	addq	%r11, %rax
	adcq	$0, %r12
	addq	%rbp, %r9
	mulxq	-72(%rsp), %r11, %r9            # 8-byte Folded Reload
	adcq	%rcx, %rax
	adcq	$0, %r12
	addq	%rsi, %r11
	adcq	$0, %r9
	addq	%r12, %r11
	adcq	$0, %r9
	mulxq	-16(%rsp), %r12, %rcx           # 8-byte Folded Reload
	addq	%r8, %r12
	adcq	$0, %rcx
	addq	%r9, %r12
	adcq	$0, %rcx
	mulxq	-80(%rsp), %r8, %rsi            # 8-byte Folded Reload
	addq	%rbx, %r8
	adcq	$0, %rsi
	addq	%rcx, %r8
	adcq	$0, %rsi
	mulxq	-88(%rsp), %rbp, %rcx           # 8-byte Folded Reload
	addq	%r14, %rbp
	adcq	$0, %rcx
	addq	%rsi, %rbp
	adcq	$0, %rcx
	mulxq	-96(%rsp), %r9, %rsi            # 8-byte Folded Reload
	addq	%r15, %r9
	adcq	$0, %rsi
	addq	%rcx, %r9
	mulxq	-24(%rsp), %r15, %rcx           # 8-byte Folded Reload
	adcq	$0, %rsi
	addq	%r10, %r15
	adcq	$0, %rcx
	addq	%rsi, %r15
	adcq	$0, %rcx
	movq	-120(%rsp), %rdx                # 8-byte Reload
	movq	32(%rdx), %rdx
	mulxq	8(%rsp), %rbx, %r10             # 8-byte Folded Reload
	mulxq	16(%rsp), %rsi, %r14            # 8-byte Folded Reload
	addb	$255, -128(%rsp)                # 1-byte Folded Spill
	adcq	%r13, %rcx
	setb	-128(%rsp)                      # 1-byte Folded Spill
	addq	%r11, %rsi
	adcq	$0, %r14
	addq	%rax, %rbx
	movq	%rbx, (%rsp)                    # 8-byte Spill
	adcq	%r10, %rsi
	mulxq	24(%rsp), %rbx, %rax            # 8-byte Folded Reload
	adcq	$0, %r14
	addq	%r12, %rbx
	adcq	$0, %rax
	addq	%r14, %rbx
	mulxq	-56(%rsp), %r14, %r10           # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r8, %r14
	adcq	$0, %r10
	addq	%rax, %r14
	mulxq	-48(%rsp), %r11, %rax           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%rbp, %r11
	adcq	$0, %rax
	mulxq	-40(%rsp), %r8, %r12            # 8-byte Folded Reload
	addq	%r10, %r11
	adcq	$0, %rax
	addq	%r9, %r8
	mulxq	-32(%rsp), %r10, %r9            # 8-byte Folded Reload
	adcq	$0, %r12
	addq	%rax, %r8
	adcq	$0, %r12
	addq	%r15, %r10
	adcq	$0, %r9
	addq	%r12, %r10
	mulxq	-112(%rsp), %r15, %r13          # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%rcx, %r15
	adcq	$0, %r13
	addq	%r9, %r15
	adcq	$0, %r13
	movq	-104(%rsp), %rdx                # 8-byte Reload
	movq	(%rsp), %rbp                    # 8-byte Reload
	imulq	%rbp, %rdx
	mulxq	-8(%rsp), %r9, %rcx             # 8-byte Folded Reload
	mulxq	-64(%rsp), %rax, %r12           # 8-byte Folded Reload
	addq	%rsi, %rax
	adcq	$0, %r12
	addq	%rbp, %r9
	adcq	%rcx, %rax
	adcq	$0, %r12
	mulxq	-72(%rsp), %r9, %rcx            # 8-byte Folded Reload
	addq	%rbx, %r9
	adcq	$0, %rcx
	addq	%r12, %r9
	adcq	$0, %rcx
	mulxq	-16(%rsp), %rsi, %r12           # 8-byte Folded Reload
	addq	%r14, %rsi
	adcq	$0, %r12
	addq	%rcx, %rsi
	mulxq	-80(%rsp), %rbx, %rcx           # 8-byte Folded Reload
	adcq	$0, %r12
	addq	%r11, %rbx
	adcq	$0, %rcx
	mulxq	-88(%rsp), %r14, %r11           # 8-byte Folded Reload
	addq	%r12, %rbx
	adcq	$0, %rcx
	addq	%r8, %r14
	adcq	$0, %r11
	addq	%rcx, %r14
	adcq	$0, %r11
	mulxq	-96(%rsp), %r12, %r8            # 8-byte Folded Reload
	addq	%r10, %r12
	adcq	$0, %r8
	addq	%r11, %r12
	adcq	$0, %r8
	mulxq	-24(%rsp), %r10, %rcx           # 8-byte Folded Reload
	addq	%r15, %r10
	adcq	$0, %rcx
	addq	%r8, %r10
	adcq	$0, %rcx
	movq	-120(%rsp), %rdx                # 8-byte Reload
	movq	40(%rdx), %rdx
	mulxq	8(%rsp), %rbp, %r11             # 8-byte Folded Reload
	addb	$255, -128(%rsp)                # 1-byte Folded Spill
	adcq	%r13, %rcx
	mulxq	16(%rsp), %r8, %r15             # 8-byte Folded Reload
	setb	-128(%rsp)                      # 1-byte Folded Spill
	addq	%r9, %r8
	adcq	$0, %r15
	addq	%rax, %rbp
	mulxq	24(%rsp), %rax, %r9             # 8-byte Folded Reload
	adcq	%r11, %r8
	adcq	$0, %r15
	addq	%rsi, %rax
	mulxq	-56(%rsp), %rsi, %r13           # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%r15, %rax
	adcq	$0, %r9
	addq	%rbx, %rsi
	adcq	$0, %r13
	addq	%r9, %rsi
	mulxq	-48(%rsp), %r11, %r9            # 8-byte Folded Reload
	adcq	$0, %r13
	addq	%r14, %r11
	adcq	$0, %r9
	addq	%r13, %r11
	mulxq	-40(%rsp), %rbx, %r15           # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%r12, %rbx
	adcq	$0, %r15
	addq	%r9, %rbx
	mulxq	-32(%rsp), %r14, %r9            # 8-byte Folded Reload
	adcq	$0, %r15
	addq	%r10, %r14
	adcq	$0, %r9
	addq	%r15, %r14
	mulxq	-112(%rsp), %r10, %r13          # 8-byte Folded Reload
	adcq	$0, %r9
	addq	%rcx, %r10
	adcq	$0, %r13
	addq	%r9, %r10
	adcq	$0, %r13
	movq	-104(%rsp), %rdx                # 8-byte Reload
	imulq	%rbp, %rdx
	mulxq	-8(%rsp), %r9, %r15             # 8-byte Folded Reload
	mulxq	-64(%rsp), %rcx, %r12           # 8-byte Folded Reload
	addq	%r8, %rcx
	adcq	$0, %r12
	addq	%rbp, %r9
	mulxq	-72(%rsp), %r9, %r8             # 8-byte Folded Reload
	adcq	%r15, %rcx
	adcq	$0, %r12
	addq	%rax, %r9
	adcq	$0, %r8
	addq	%r12, %r9
	adcq	$0, %r8
	mulxq	-16(%rsp), %r12, %rax           # 8-byte Folded Reload
	addq	%rsi, %r12
	adcq	$0, %rax
	addq	%r8, %r12
	adcq	$0, %rax
	mulxq	-80(%rsp), %rbp, %rsi           # 8-byte Folded Reload
	addq	%r11, %rbp
	adcq	$0, %rsi
	addq	%rax, %rbp
	adcq	$0, %rsi
	mulxq	-88(%rsp), %r15, %rax           # 8-byte Folded Reload
	addq	%rbx, %r15
	adcq	$0, %rax
	addq	%rsi, %r15
	adcq	$0, %rax
	mulxq	-96(%rsp), %r11, %rbx           # 8-byte Folded Reload
	addq	%r14, %r11
	adcq	$0, %rbx
	addq	%rax, %r11
	mulxq	-24(%rsp), %rsi, %r8            # 8-byte Folded Reload
	adcq	$0, %rbx
	addq	%r10, %rsi
	adcq	$0, %r8
	addq	%rbx, %rsi
	adcq	$0, %r8
	movq	-120(%rsp), %rax                # 8-byte Reload
	movq	48(%rax), %rdx
	mulxq	8(%rsp), %r14, %rbx             # 8-byte Folded Reload
	mulxq	16(%rsp), %r10, %rax            # 8-byte Folded Reload
	addb	$255, -128(%rsp)                # 1-byte Folded Spill
	adcq	%r13, %r8
	setb	-128(%rsp)                      # 1-byte Folded Spill
	addq	%r9, %r10
	adcq	$0, %rax
	addq	%rcx, %r14
	movq	%r14, %r13
	adcq	%rbx, %r10
	mulxq	24(%rsp), %r14, %rcx            # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r12, %r14
	adcq	$0, %rcx
	addq	%rax, %r14
	mulxq	-56(%rsp), %rbx, %rax           # 8-byte Folded Reload
	adcq	$0, %rcx
	addq	%rbp, %rbx
	adcq	$0, %rax
	addq	%rcx, %rbx
	mulxq	-48(%rsp), %rcx, %r12           # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r15, %rcx
	adcq	$0, %r12
	mulxq	-40(%rsp), %r9, %r15            # 8-byte Folded Reload
	addq	%rax, %rcx
	adcq	$0, %r12
	addq	%r11, %r9
	mulxq	-32(%rsp), %r11, %rax           # 8-byte Folded Reload
	adcq	$0, %r15
	addq	%r12, %r9
	adcq	$0, %r15
	addq	%rsi, %r11
	adcq	$0, %rax
	addq	%r15, %r11
	mulxq	-112(%rsp), %r15, %rbp          # 8-byte Folded Reload
	adcq	$0, %rax
	addq	%r8, %r15
	adcq	$0, %rbp
	addq	%rax, %r15
	adcq	$0, %rbp
	movq	-104(%rsp), %rdx                # 8-byte Reload
	imulq	%r13, %rdx
	mulxq	-8(%rsp), %r8, %rax             # 8-byte Folded Reload
	mulxq	-64(%rsp), %rsi, %r12           # 8-byte Folded Reload
	addq	%r10, %rsi
	adcq	$0, %r12
	addq	%r13, %r8
	adcq	%rax, %rsi
	movq	%rsi, (%rsp)                    # 8-byte Spill
	adcq	$0, %r12
	mulxq	-72(%rsp), %rax, %r8            # 8-byte Folded Reload
	addq	%r14, %rax
	adcq	$0, %r8
	addq	%r12, %rax
	mulxq	-16(%rsp), %r10, %r14           # 8-byte Folded Reload
	adcq	$0, %r8
	addq	%rbx, %r10
	adcq	$0, %r14
	mulxq	-80(%rsp), %rbx, %r12           # 8-byte Folded Reload
	addq	%r8, %r10
	adcq	$0, %r14
	addq	%rcx, %rbx
	adcq	$0, %r12
	addq	%r14, %rbx
	adcq	$0, %r12
	mulxq	-88(%rsp), %r14, %rcx           # 8-byte Folded Reload
	addq	%r9, %r14
	adcq	$0, %rcx
	addq	%r12, %r14
	adcq	$0, %rcx
	mulxq	-96(%rsp), %r12, %r8            # 8-byte Folded Reload
	addq	%r11, %r12
	adcq	$0, %r8
	addq	%rcx, %r12
	adcq	$0, %r8
	mulxq	-24(%rsp), %r13, %r9            # 8-byte Folded Reload
	addq	%r15, %r13
	adcq	$0, %r9
	addq	%r8, %r13
	adcq	$0, %r9
	addb	$255, -128(%rsp)                # 1-byte Folded Spill
	adcq	%rbp, %r9
	movq	-120(%rsp), %rcx                # 8-byte Reload
	movq	56(%rcx), %rdx
	mulxq	8(%rsp), %rbp, %rcx             # 8-byte Folded Reload
	mulxq	16(%rsp), %r8, %r11             # 8-byte Folded Reload
	setb	-120(%rsp)                      # 1-byte Folded Spill
	addq	%rax, %r8
	adcq	$0, %r11
	mulxq	24(%rsp), %r15, %rsi            # 8-byte Folded Reload
	addq	(%rsp), %rbp                    # 8-byte Folded Reload
	adcq	%rcx, %r8
	adcq	$0, %r11
	addq	%r10, %r15
	adcq	$0, %rsi
	addq	%r11, %r15
	mulxq	-56(%rsp), %r11, %r10           # 8-byte Folded Reload
	adcq	$0, %rsi
	addq	%rbx, %r11
	adcq	$0, %r10
	addq	%rsi, %r11
	mulxq	-48(%rsp), %rax, %rbx           # 8-byte Folded Reload
	adcq	$0, %r10
	addq	%r14, %rax
	adcq	$0, %rbx
	addq	%r10, %rax
	movq	%rax, -48(%rsp)                 # 8-byte Spill
	mulxq	-40(%rsp), %rsi, %r14           # 8-byte Folded Reload
	adcq	$0, %rbx
	addq	%r12, %rsi
	adcq	$0, %r14
	addq	%rbx, %rsi
	mulxq	-32(%rsp), %r10, %r12           # 8-byte Folded Reload
	adcq	$0, %r14
	addq	%r13, %r10
	adcq	$0, %r12
	mulxq	-112(%rsp), %rbx, %r13          # 8-byte Folded Reload
	addq	%r14, %r10
	adcq	$0, %r12
	addq	%r9, %rbx
	adcq	$0, %r13
	addq	%r12, %rbx
	adcq	$0, %r13
	movzbl	-120(%rsp), %edx                # 1-byte Folded Reload
	addb	$255, %dl
	movq	%r13, %r14
	adcq	$0, %r14
	movq	-104(%rsp), %rdx                # 8-byte Reload
	imulq	%rbp, %rdx
	mulxq	-8(%rsp), %rax, %rcx            # 8-byte Folded Reload
	movq	%rdi, -32(%rsp)                 # 8-byte Spill
	movq	%rdx, %r9
	mulxq	-64(%rsp), %rdx, %r12           # 8-byte Folded Reload
	addq	%r8, %rdx
	adcq	$0, %r12
	addq	%rbp, %rax
	adcq	%rcx, %rdx
	movq	%rdx, -104(%rsp)                # 8-byte Spill
	adcq	$0, %r12
	movq	%r9, %rdx
	mulxq	-72(%rsp), %rax, %rcx           # 8-byte Folded Reload
	addq	%r15, %rax
	adcq	$0, %rcx
	addq	%r12, %rax
	movq	%rax, -112(%rsp)                # 8-byte Spill
	mulxq	-16(%rsp), %r12, %r15           # 8-byte Folded Reload
	adcq	$0, %rcx
	addq	%r11, %r12
	adcq	$0, %r15
	mulxq	-80(%rsp), %rbp, %r11           # 8-byte Folded Reload
	addq	%rcx, %r12
	movq	%r12, -56(%rsp)                 # 8-byte Spill
	adcq	$0, %r15
	addq	-48(%rsp), %rbp                 # 8-byte Folded Reload
	adcq	$0, %r11
	addq	%r15, %rbp
	adcq	$0, %r11
	mulxq	-88(%rsp), %r8, %r15            # 8-byte Folded Reload
	addq	%rsi, %r8
	adcq	$0, %r15
	addq	%r11, %r8
	adcq	$0, %r15
	mulxq	-96(%rsp), %rdi, %r11           # 8-byte Folded Reload
	addq	%r10, %rdi
	adcq	$0, %r11
	addq	%r15, %rdi
	adcq	$0, %r11
	movq	-24(%rsp), %r9                  # 8-byte Reload
	mulxq	%r9, %rax, %r15
	addq	%rbx, %rax
	adcq	$0, %r15
	addq	%r11, %rax
	adcq	$0, %r15
	addb	$255, -120(%rsp)                # 1-byte Folded Spill
	movq	%r15, %rcx
	adcq	%r13, %rcx
	cmpq	%r13, %r14
	setae	-40(%rsp)                       # 1-byte Folded Spill
	cmpq	%r15, %rcx
	setae	%sil
	movq	-104(%rsp), %r11                # 8-byte Reload
	subq	-8(%rsp), %r11                  # 8-byte Folded Reload
	movq	-112(%rsp), %rbx                # 8-byte Reload
	sbbq	-64(%rsp), %rbx                 # 8-byte Folded Reload
	movq	%r12, %r14
	sbbq	-72(%rsp), %r14                 # 8-byte Folded Reload
	movq	%rbp, %r15
	sbbq	-16(%rsp), %r15                 # 8-byte Folded Reload
	movq	%r8, %r12
	sbbq	-80(%rsp), %r12                 # 8-byte Folded Reload
	movq	%rdi, %r13
	sbbq	-88(%rsp), %r13                 # 8-byte Folded Reload
	movq	%rax, %rdx
	sbbq	-96(%rsp), %rdx                 # 8-byte Folded Reload
	movq	%rcx, %r10
	sbbq	%r9, %r10
	setb	%r9b
	andb	-40(%rsp), %sil                 # 1-byte Folded Reload
	testb	%r9b, %sil
	cmovneq	-104(%rsp), %r11                # 8-byte Folded Reload
	cmovneq	-112(%rsp), %rbx                # 8-byte Folded Reload
	cmovneq	-56(%rsp), %r14                 # 8-byte Folded Reload
	cmovneq	%rbp, %r15
	cmovneq	%r8, %r12
	cmovneq	%rdi, %r13
	cmovneq	%rax, %rdx
	cmovneq	%rcx, %r10
	movq	-32(%rsp), %rax                 # 8-byte Reload
	movq	%r11, (%rax)
	movq	%rbx, 8(%rax)
	movq	%r14, 16(%rax)
	movq	%r15, 24(%rax)
	movq	%r12, 32(%rax)
	movq	%r13, 40(%rax)
	movq	%rdx, 48(%rax)
	movq	%r10, 56(%rax)
	addq	$32, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
.Lfunc_end7:
	.size	zfactor_montmul8, .Lfunc_end7-zfactor_montmul8
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
