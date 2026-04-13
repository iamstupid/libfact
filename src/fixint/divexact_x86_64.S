	.text
	.file	"divexact_ir.ll"
	.globl	divexact1_ip_peel_2             # -- Begin function divexact1_ip_peel_2
	.p2align	4, 0x90
	.type	divexact1_ip_peel_2,@function
divexact1_ip_peel_2:                    # @divexact1_ip_peel_2
# %bb.0:                                # %entry
	movq	%rdx, %r8
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_1:                                # %loop
                                        # =>This Inner Loop Header: Depth=1
	movq	(%rdi), %rdx
	imulq	%rsi, %rdx
	mulxq	%r8, %r10, %r10
	movq	8(%rdi), %r11
	movq	%r11, %r9
	subq	%r10, %r9
	imulq	%rsi, %r9
	cmpq	%r10, %r11
	movq	%rdx, (%rdi)
	movq	%r9, 8(%rdi)
	movq	%r9, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	jne	.LBB0_3
# %bb.2:                                # %loop_continue
                                        #   in Loop: Header=BB0_1 Depth=1
	movq	(%rdi), %rdx
	movq	%rdx, (%rcx)
	movq	8(%rdi), %rdx
	movq	%rdx, 8(%rcx)
	incl	%eax
	jmp	.LBB0_1
.LBB0_3:                                # %loop_exit
	testl	%eax, %eax
	je	.LBB0_5
# %bb.4:                                # %do_restore
	vmovups	(%rcx), %xmm0
	vmovups	%xmm0, (%rdi)
.LBB0_5:                                # %done
	retq
.Lfunc_end0:
	.size	divexact1_ip_peel_2, .Lfunc_end0-divexact1_ip_peel_2
                                        # -- End function
	.globl	divexact1_ip_peel_3             # -- Begin function divexact1_ip_peel_3
	.p2align	4, 0x90
	.type	divexact1_ip_peel_3,@function
divexact1_ip_peel_3:                    # @divexact1_ip_peel_3
# %bb.0:                                # %entry
	pushq	%rbx
	movq	%rdx, %r8
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB1_1:                                # %loop
                                        # =>This Inner Loop Header: Depth=1
	movq	(%rdi), %r9
	imulq	%rsi, %r9
	movq	%r9, %rdx
	mulxq	%r8, %r10, %r10
	movq	8(%rdi), %r11
	movq	16(%rdi), %rbx
	movq	%r11, %rdx
	subq	%r10, %rdx
	imulq	%rsi, %rdx
	cmpq	%r10, %r11
	mulxq	%r8, %r11, %r11
	adcq	$0, %r11
	movq	%rbx, %r10
	subq	%r11, %r10
	imulq	%rsi, %r10
	cmpq	%r11, %rbx
	movq	%r9, (%rdi)
	movq	%rdx, 8(%rdi)
	movq	%r10, 16(%rdi)
	movq	%r10, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	jne	.LBB1_3
# %bb.2:                                # %loop_continue
                                        #   in Loop: Header=BB1_1 Depth=1
	movq	16(%rdi), %rdx
	movq	%rdx, 16(%rcx)
	movq	(%rdi), %rdx
	movq	%rdx, (%rcx)
	movq	8(%rdi), %rdx
	movq	%rdx, 8(%rcx)
	incl	%eax
	jmp	.LBB1_1
.LBB1_3:                                # %loop_exit
	testl	%eax, %eax
	je	.LBB1_5
# %bb.4:                                # %do_restore
	movq	16(%rcx), %rdx
	movq	%rdx, 16(%rdi)
	vmovups	(%rcx), %xmm0
	vmovups	%xmm0, (%rdi)
.LBB1_5:                                # %done
	popq	%rbx
	retq
.Lfunc_end1:
	.size	divexact1_ip_peel_3, .Lfunc_end1-divexact1_ip_peel_3
                                        # -- End function
	.globl	divexact1_ip_peel_4             # -- Begin function divexact1_ip_peel_4
	.p2align	4, 0x90
	.type	divexact1_ip_peel_4,@function
divexact1_ip_peel_4:                    # @divexact1_ip_peel_4
# %bb.0:                                # %entry
	pushq	%r14
	pushq	%rbx
	movq	%rdx, %r8
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB2_1:                                # %loop
                                        # =>This Inner Loop Header: Depth=1
	movq	(%rdi), %r9
	imulq	%rsi, %r9
	movq	%r9, %rdx
	mulxq	%r8, %rdx, %rdx
	movq	8(%rdi), %r11
	movq	16(%rdi), %rbx
	movq	%r11, %r10
	subq	%rdx, %r10
	imulq	%rsi, %r10
	cmpq	%rdx, %r11
	movq	%r10, %rdx
	mulxq	%r8, %r11, %r11
	adcq	$0, %r11
	movq	%rbx, %rdx
	subq	%r11, %rdx
	imulq	%rsi, %rdx
	cmpq	%r11, %rbx
	mulxq	%r8, %rbx, %rbx
	adcq	$0, %rbx
	movq	24(%rdi), %r14
	movq	%r14, %r11
	subq	%rbx, %r11
	imulq	%rsi, %r11
	cmpq	%rbx, %r14
	movq	%r9, (%rdi)
	movq	%r10, 8(%rdi)
	movq	%rdx, 16(%rdi)
	movq	%r11, 24(%rdi)
	movq	%r11, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	jne	.LBB2_3
# %bb.2:                                # %loop_continue
                                        #   in Loop: Header=BB2_1 Depth=1
	movq	(%rdi), %rdx
	movq	%rdx, (%rcx)
	movq	8(%rdi), %rdx
	movq	%rdx, 8(%rcx)
	movq	16(%rdi), %rdx
	movq	%rdx, 16(%rcx)
	movq	24(%rdi), %rdx
	movq	%rdx, 24(%rcx)
	incl	%eax
	jmp	.LBB2_1
.LBB2_3:                                # %loop_exit
	testl	%eax, %eax
	je	.LBB2_5
# %bb.4:                                # %do_restore
	vmovups	(%rcx), %ymm0
	vmovups	%ymm0, (%rdi)
.LBB2_5:                                # %done
	popq	%rbx
	popq	%r14
	vzeroupper
	retq
.Lfunc_end2:
	.size	divexact1_ip_peel_4, .Lfunc_end2-divexact1_ip_peel_4
                                        # -- End function
	.globl	divexact1_ip_peel_5             # -- Begin function divexact1_ip_peel_5
	.p2align	4, 0x90
	.type	divexact1_ip_peel_5,@function
divexact1_ip_peel_5:                    # @divexact1_ip_peel_5
# %bb.0:                                # %entry
	pushq	%r15
	pushq	%r14
	pushq	%rbx
	movq	%rdx, %r8
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB3_1:                                # %loop
                                        # =>This Inner Loop Header: Depth=1
	movq	(%rdi), %r9
	imulq	%rsi, %r9
	movq	%r9, %rdx
	mulxq	%r8, %rdx, %rdx
	movq	8(%rdi), %r11
	movq	16(%rdi), %rbx
	movq	%r11, %r10
	subq	%rdx, %r10
	imulq	%rsi, %r10
	cmpq	%rdx, %r11
	movq	%r10, %rdx
	mulxq	%r8, %r14, %r14
	adcq	$0, %r14
	movq	%rbx, %r11
	subq	%r14, %r11
	imulq	%rsi, %r11
	movq	%r11, %rdx
	mulxq	%r8, %r15, %r15
	cmpq	%r14, %rbx
	adcq	$0, %r15
	movq	24(%rdi), %rbx
	movq	%rbx, %rdx
	subq	%r15, %rdx
	imulq	%rsi, %rdx
	mulxq	%r8, %r14, %r14
	cmpq	%r15, %rbx
	adcq	$0, %r14
	movq	32(%rdi), %r15
	movq	%r15, %rbx
	subq	%r14, %rbx
	imulq	%rsi, %rbx
	cmpq	%r14, %r15
	movq	%r9, (%rdi)
	movq	%r10, 8(%rdi)
	movq	%r11, 16(%rdi)
	movq	%rdx, 24(%rdi)
	movq	%rbx, 32(%rdi)
	movq	%rbx, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	jne	.LBB3_3
# %bb.2:                                # %loop_continue
                                        #   in Loop: Header=BB3_1 Depth=1
	movq	32(%rdi), %rdx
	movq	%rdx, 32(%rcx)
	movq	(%rdi), %rdx
	movq	%rdx, (%rcx)
	movq	8(%rdi), %rdx
	movq	%rdx, 8(%rcx)
	movq	16(%rdi), %rdx
	movq	%rdx, 16(%rcx)
	movq	24(%rdi), %rdx
	movq	%rdx, 24(%rcx)
	incl	%eax
	jmp	.LBB3_1
.LBB3_3:                                # %loop_exit
	testl	%eax, %eax
	je	.LBB3_5
# %bb.4:                                # %do_restore
	movq	32(%rcx), %rdx
	movq	%rdx, 32(%rdi)
	vmovups	(%rcx), %ymm0
	vmovups	%ymm0, (%rdi)
.LBB3_5:                                # %done
	popq	%rbx
	popq	%r14
	popq	%r15
	vzeroupper
	retq
.Lfunc_end3:
	.size	divexact1_ip_peel_5, .Lfunc_end3-divexact1_ip_peel_5
                                        # -- End function
	.globl	divexact1_ip_peel_6             # -- Begin function divexact1_ip_peel_6
	.p2align	4, 0x90
	.type	divexact1_ip_peel_6,@function
divexact1_ip_peel_6:                    # @divexact1_ip_peel_6
# %bb.0:                                # %entry
	pushq	%r15
	pushq	%r14
	pushq	%r12
	pushq	%rbx
	movq	%rdx, %r8
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB4_1:                                # %loop
                                        # =>This Inner Loop Header: Depth=1
	movq	(%rdi), %r9
	imulq	%rsi, %r9
	movq	%r9, %rdx
	mulxq	%r8, %r11, %r11
	movq	8(%rdi), %rbx
	movq	16(%rdi), %r14
	movq	%rbx, %r10
	subq	%r11, %r10
	imulq	%rsi, %r10
	movq	%r10, %rdx
	mulxq	%r8, %rdx, %rdx
	cmpq	%r11, %rbx
	adcq	$0, %rdx
	movq	%r14, %r11
	subq	%rdx, %r11
	imulq	%rsi, %r11
	cmpq	%rdx, %r14
	movq	%r11, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	movq	24(%rdi), %r14
	movq	%r14, %rbx
	subq	%rdx, %rbx
	imulq	%rsi, %rbx
	cmpq	%rdx, %r14
	movq	%rbx, %rdx
	mulxq	%r8, %r14, %r14
	adcq	$0, %r14
	movq	32(%rdi), %r15
	movq	%r15, %rdx
	subq	%r14, %rdx
	imulq	%rsi, %rdx
	cmpq	%r14, %r15
	mulxq	%r8, %r15, %r15
	adcq	$0, %r15
	movq	40(%rdi), %r12
	movq	%r12, %r14
	subq	%r15, %r14
	imulq	%rsi, %r14
	cmpq	%r15, %r12
	movq	%r9, (%rdi)
	movq	%r10, 8(%rdi)
	movq	%r11, 16(%rdi)
	movq	%rbx, 24(%rdi)
	movq	%rdx, 32(%rdi)
	movq	%r14, 40(%rdi)
	movq	%r14, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	jne	.LBB4_3
# %bb.2:                                # %loop_continue
                                        #   in Loop: Header=BB4_1 Depth=1
	movq	(%rdi), %rdx
	movq	8(%rdi), %r9
	movq	16(%rdi), %r10
	movq	24(%rdi), %r11
	movq	16(%rdi), %rbx
	movq	%rbx, 16(%rcx)
	movq	24(%rdi), %rbx
	movq	%rbx, 24(%rcx)
	movq	32(%rdi), %rbx
	movq	%rbx, 32(%rcx)
	movq	40(%rdi), %rbx
	movq	%rbx, 40(%rcx)
	movq	%rdx, (%rcx)
	movq	%r9, 8(%rcx)
	movq	%r10, 16(%rcx)
	movq	%r11, 24(%rcx)
	incl	%eax
	jmp	.LBB4_1
.LBB4_3:                                # %loop_exit
	testl	%eax, %eax
	je	.LBB4_5
# %bb.4:                                # %do_restore
	vmovups	(%rcx), %ymm0
	vmovups	16(%rcx), %ymm1
	vmovups	%ymm1, 16(%rdi)
	vmovups	%ymm0, (%rdi)
.LBB4_5:                                # %done
	popq	%rbx
	popq	%r12
	popq	%r14
	popq	%r15
	vzeroupper
	retq
.Lfunc_end4:
	.size	divexact1_ip_peel_6, .Lfunc_end4-divexact1_ip_peel_6
                                        # -- End function
	.globl	divexact1_ip_peel_7             # -- Begin function divexact1_ip_peel_7
	.p2align	4, 0x90
	.type	divexact1_ip_peel_7,@function
divexact1_ip_peel_7:                    # @divexact1_ip_peel_7
# %bb.0:                                # %entry
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	movq	%rdx, %r8
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB5_1:                                # %loop
                                        # =>This Inner Loop Header: Depth=1
	movq	(%rdi), %r9
	imulq	%rsi, %r9
	movq	%r9, %rdx
	mulxq	%r8, %rdx, %rdx
	movq	8(%rdi), %r11
	movq	16(%rdi), %rbx
	movq	%r11, %r10
	subq	%rdx, %r10
	imulq	%rsi, %r10
	cmpq	%rdx, %r11
	movq	%r10, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	movq	%rbx, %r11
	subq	%rdx, %r11
	imulq	%rsi, %r11
	cmpq	%rdx, %rbx
	movq	%r11, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	movq	24(%rdi), %r14
	movq	%r14, %rbx
	subq	%rdx, %rbx
	imulq	%rsi, %rbx
	cmpq	%rdx, %r14
	movq	%rbx, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	movq	32(%rdi), %r15
	movq	%r15, %r14
	subq	%rdx, %r14
	imulq	%rsi, %r14
	cmpq	%rdx, %r15
	movq	%r14, %rdx
	mulxq	%r8, %r15, %r15
	adcq	$0, %r15
	movq	40(%rdi), %r12
	movq	%r12, %rdx
	subq	%r15, %rdx
	imulq	%rsi, %rdx
	cmpq	%r15, %r12
	mulxq	%r8, %r12, %r12
	adcq	$0, %r12
	movq	48(%rdi), %r13
	movq	%r13, %r15
	subq	%r12, %r15
	imulq	%rsi, %r15
	cmpq	%r12, %r13
	movq	%r9, (%rdi)
	movq	%r10, 8(%rdi)
	movq	%r11, 16(%rdi)
	movq	%rbx, 24(%rdi)
	movq	%r14, 32(%rdi)
	movq	%rdx, 40(%rdi)
	movq	%r15, 48(%rdi)
	movq	%r15, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	jne	.LBB5_3
# %bb.2:                                # %loop_continue
                                        #   in Loop: Header=BB5_1 Depth=1
	movq	(%rdi), %rdx
	movq	8(%rdi), %r9
	movq	16(%rdi), %r10
	movq	24(%rdi), %r11
	movq	24(%rdi), %rbx
	movq	%rbx, 24(%rcx)
	movq	32(%rdi), %rbx
	movq	%rbx, 32(%rcx)
	movq	40(%rdi), %rbx
	movq	%rbx, 40(%rcx)
	movq	48(%rdi), %rbx
	movq	%rbx, 48(%rcx)
	movq	%rdx, (%rcx)
	movq	%r9, 8(%rcx)
	movq	%r10, 16(%rcx)
	movq	%r11, 24(%rcx)
	incl	%eax
	jmp	.LBB5_1
.LBB5_3:                                # %loop_exit
	testl	%eax, %eax
	je	.LBB5_5
# %bb.4:                                # %do_restore
	vmovups	(%rcx), %ymm0
	vmovups	24(%rcx), %ymm1
	vmovups	%ymm1, 24(%rdi)
	vmovups	%ymm0, (%rdi)
.LBB5_5:                                # %done
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	vzeroupper
	retq
.Lfunc_end5:
	.size	divexact1_ip_peel_7, .Lfunc_end5-divexact1_ip_peel_7
                                        # -- End function
	.globl	divexact1_ip_peel_8             # -- Begin function divexact1_ip_peel_8
	.p2align	4, 0x90
	.type	divexact1_ip_peel_8,@function
divexact1_ip_peel_8:                    # @divexact1_ip_peel_8
# %bb.0:                                # %entry
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	movq	%rdx, %r8
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB6_1:                                # %loop
                                        # =>This Inner Loop Header: Depth=1
	movq	(%rdi), %r9
	imulq	%rsi, %r9
	movq	%r9, %rdx
	mulxq	%r8, %rdx, %rdx
	movq	8(%rdi), %r11
	movq	16(%rdi), %rbx
	movq	%r11, %r10
	subq	%rdx, %r10
	imulq	%rsi, %r10
	cmpq	%rdx, %r11
	movq	%r10, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	movq	%rbx, %r11
	subq	%rdx, %r11
	imulq	%rsi, %r11
	cmpq	%rdx, %rbx
	movq	%r11, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	movq	24(%rdi), %r14
	movq	%r14, %rbx
	subq	%rdx, %rbx
	imulq	%rsi, %rbx
	cmpq	%rdx, %r14
	movq	%rbx, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	movq	32(%rdi), %r15
	movq	%r15, %r14
	subq	%rdx, %r14
	imulq	%rsi, %r14
	cmpq	%rdx, %r15
	movq	%r14, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	movq	40(%rdi), %r12
	movq	%r12, %r15
	subq	%rdx, %r15
	imulq	%rsi, %r15
	cmpq	%rdx, %r12
	movq	%r15, %rdx
	mulxq	%r8, %r12, %r12
	adcq	$0, %r12
	movq	48(%rdi), %r13
	movq	%r13, %rdx
	subq	%r12, %rdx
	imulq	%rsi, %rdx
	cmpq	%r12, %r13
	mulxq	%r8, %r13, %r13
	adcq	$0, %r13
	movq	56(%rdi), %rbp
	movq	%rbp, %r12
	subq	%r13, %r12
	imulq	%rsi, %r12
	cmpq	%r13, %rbp
	movq	%r9, (%rdi)
	movq	%r10, 8(%rdi)
	movq	%r11, 16(%rdi)
	movq	%rbx, 24(%rdi)
	movq	%r14, 32(%rdi)
	movq	%r15, 40(%rdi)
	movq	%rdx, 48(%rdi)
	movq	%r12, 56(%rdi)
	movq	%r12, %rdx
	mulxq	%r8, %rdx, %rdx
	adcq	$0, %rdx
	jne	.LBB6_3
# %bb.2:                                # %loop_continue
                                        #   in Loop: Header=BB6_1 Depth=1
	movq	(%rdi), %rdx
	movq	8(%rdi), %r9
	movq	16(%rdi), %r10
	movq	24(%rdi), %r11
	movq	32(%rdi), %rbx
	movq	%rbx, 32(%rcx)
	movq	40(%rdi), %rbx
	movq	%rbx, 40(%rcx)
	movq	48(%rdi), %rbx
	movq	%rbx, 48(%rcx)
	movq	56(%rdi), %rbx
	movq	%rbx, 56(%rcx)
	movq	%rdx, (%rcx)
	movq	%r9, 8(%rcx)
	movq	%r10, 16(%rcx)
	movq	%r11, 24(%rcx)
	incl	%eax
	jmp	.LBB6_1
.LBB6_3:                                # %loop_exit
	testl	%eax, %eax
	je	.LBB6_5
# %bb.4:                                # %do_restore
	vmovups	(%rcx), %ymm0
	vmovups	32(%rcx), %ymm1
	vmovups	%ymm1, 32(%rdi)
	vmovups	%ymm0, (%rdi)
.LBB6_5:                                # %done
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	retq
.Lfunc_end6:
	.size	divexact1_ip_peel_8, .Lfunc_end6-divexact1_ip_peel_8
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
