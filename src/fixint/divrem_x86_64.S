	.file	"divrem_ir.ll"
	.text
	.globl	zfactor_mg_divrem               # -- Begin function zfactor_mg_divrem
	.p2align	4
	.type	zfactor_mg_divrem,@function
zfactor_mg_divrem:                      # @zfactor_mg_divrem
# %bb.0:                                # %entry
	pushq	%rbp
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$104, %rsp
	movq	%r9, %r14
	movq	%rdx, %r12
	movq	%rsi, %rbx
	movq	%rdi, -88(%rbp)                 # 8-byte Spill
	leaq	-1(%r9), %rax
	lzcntq	(%r8,%rax,8), %r15
	movl	$64, %r10d
	subq	%r15, %r10
	movq	%rsp, %rsi
	leaq	31(,%rcx,8), %rax
	andq	$-16, %rax
	subq	%rax, %rsi
	movq	%rsi, %rsp
	movq	%rsp, %r13
	leaq	15(,%r9,8), %rax
	andq	$-16, %rax
	subq	%rax, %r13
	movq	%r13, %rsp
	movq	%r9, %rax
	subq	$1, %rax
	movq	%rax, -144(%rbp)                # 8-byte Spill
	movq	%rsi, -48(%rbp)                 # 8-byte Spill
	movq	%r10, -72(%rbp)                 # 8-byte Spill
	jae	.LBB0_3
# %bb.1:                                # %ushift_loop.preheader
	testq	%rcx, %rcx
	je	.LBB0_65
# %bb.2:                                # %ushift_body.lr.ph
	testq	%r15, %r15
	jne	.LBB0_15
	jmp	.LBB0_8
.LBB0_3:                                # %vshift_body.lr.ph
	testq	%r15, %r15
	je	.LBB0_7
# %bb.4:                                # %vshift_body.preheader
	cmpq	$16, %r14
	jae	.LBB0_9
# %bb.5:
	xorl	%edx, %edx
	xorl	%eax, %eax
	jmp	.LBB0_13
.LBB0_7:                                # %ushift_loop.preheader.thread36
	leaq	(,%r14,8), %rdx
	movq	%r13, %rdi
	movq	%r8, %rsi
	movq	%rcx, -56(%rbp)                 # 8-byte Spill
	callq	memcpy@PLT
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	-48(%rbp), %rsi                 # 8-byte Reload
	movq	-72(%rbp), %r10                 # 8-byte Reload
	testq	%rcx, %rcx
	je	.LBB0_65
.LBB0_8:                                # %ushift_body.us.preheader
	movq	%rcx, -56(%rbp)                 # 8-byte Spill
	leaq	(,%rcx,8), %rdx
	movq	%rsi, %rdi
	movq	%r12, %rsi
	callq	memcpy@PLT
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	-48(%rbp), %rsi                 # 8-byte Reload
	movq	-72(%rbp), %r10                 # 8-byte Reload
	jmp	.LBB0_22
.LBB0_9:                                # %vector.ph
	movq	%r14, %rax
	andq	$-16, %rax
	vmovq	%r15, %xmm0
	vpbroadcastq	%xmm0, %ymm0
	vmovq	%r10, %xmm1
	vpbroadcastq	%xmm1, %ymm1
	vpxor	%xmm2, %xmm2, %xmm2
	xorl	%edx, %edx
	.p2align	4
.LBB0_10:                               # %vector.body
                                        # =>This Inner Loop Header: Depth=1
	vmovdqu	(%r8,%rdx,8), %ymm3
	vmovdqu	32(%r8,%rdx,8), %ymm4
	vmovdqu	64(%r8,%rdx,8), %ymm5
	vpblendd	$192, %ymm2, %ymm3, %ymm6       # ymm6 = ymm3[0,1,2,3,4,5],ymm2[6,7]
	vmovdqu	96(%r8,%rdx,8), %ymm2
	vpermq	$147, %ymm6, %ymm6              # ymm6 = ymm6[3,0,1,2]
	vpblendd	$192, %ymm3, %ymm4, %ymm7       # ymm7 = ymm4[0,1,2,3,4,5],ymm3[6,7]
	vpermq	$147, %ymm7, %ymm7              # ymm7 = ymm7[3,0,1,2]
	vpblendd	$192, %ymm4, %ymm5, %ymm8       # ymm8 = ymm5[0,1,2,3,4,5],ymm4[6,7]
	vpermq	$147, %ymm8, %ymm8              # ymm8 = ymm8[3,0,1,2]
	vpblendd	$192, %ymm5, %ymm2, %ymm9       # ymm9 = ymm2[0,1,2,3,4,5],ymm5[6,7]
	vpermq	$147, %ymm9, %ymm9              # ymm9 = ymm9[3,0,1,2]
	vpsllvq	%ymm0, %ymm3, %ymm3
	vpsllvq	%ymm0, %ymm4, %ymm4
	vpsllvq	%ymm0, %ymm5, %ymm5
	vpsllvq	%ymm0, %ymm2, %ymm10
	vpsrlvq	%ymm1, %ymm6, %ymm6
	vpor	%ymm6, %ymm3, %ymm3
	vpsrlvq	%ymm1, %ymm7, %ymm6
	vpor	%ymm6, %ymm4, %ymm4
	vpsrlvq	%ymm1, %ymm8, %ymm6
	vpor	%ymm6, %ymm5, %ymm5
	vpsrlvq	%ymm1, %ymm9, %ymm6
	vpor	%ymm6, %ymm10, %ymm6
	vmovdqu	%ymm3, (%r13,%rdx,8)
	vmovdqu	%ymm4, 32(%r13,%rdx,8)
	vmovdqu	%ymm5, 64(%r13,%rdx,8)
	vmovdqu	%ymm6, 96(%r13,%rdx,8)
	addq	$16, %rdx
	cmpq	%rdx, %rax
	jne	.LBB0_10
# %bb.11:                               # %middle.block
	cmpq	%rax, %r14
	je	.LBB0_14
# %bb.12:
	vextracti128	$1, %ymm2, %xmm0
	vpextrq	$1, %xmm0, %rdx
	.p2align	4
.LBB0_13:                               # %vshift_body
                                        # =>This Inner Loop Header: Depth=1
	shrxq	%r10, %rdx, %r9
	movq	(%r8,%rax,8), %rdx
	shlxq	%r15, %rdx, %rdi
	orq	%rdi, %r9
	movq	%r9, (%r13,%rax,8)
	incq	%rax
	cmpq	%rax, %r14
	jne	.LBB0_13
.LBB0_14:                               # %ushift_loop.preheader.thread
	testq	%rcx, %rcx
	je	.LBB0_65
.LBB0_15:                               # %ushift_body.preheader
	cmpq	$16, %rcx
	jae	.LBB0_17
# %bb.16:
	xorl	%edx, %edx
	xorl	%eax, %eax
	jmp	.LBB0_21
.LBB0_65:
	movb	$1, %dil
	jmp	.LBB0_23
.LBB0_17:                               # %vector.ph52
	movq	%rcx, %rax
	andq	$-16, %rax
	vmovq	%r15, %xmm0
	vpbroadcastq	%xmm0, %ymm0
	vmovq	%r10, %xmm1
	vpbroadcastq	%xmm1, %ymm1
	vpxor	%xmm2, %xmm2, %xmm2
	xorl	%edx, %edx
	.p2align	4
.LBB0_18:                               # %vector.body59
                                        # =>This Inner Loop Header: Depth=1
	vmovdqu	(%r12,%rdx,8), %ymm3
	vmovdqu	32(%r12,%rdx,8), %ymm4
	vmovdqu	64(%r12,%rdx,8), %ymm5
	vpblendd	$192, %ymm2, %ymm3, %ymm6       # ymm6 = ymm3[0,1,2,3,4,5],ymm2[6,7]
	vmovdqu	96(%r12,%rdx,8), %ymm2
	vpermq	$147, %ymm6, %ymm6              # ymm6 = ymm6[3,0,1,2]
	vpblendd	$192, %ymm3, %ymm4, %ymm7       # ymm7 = ymm4[0,1,2,3,4,5],ymm3[6,7]
	vpermq	$147, %ymm7, %ymm7              # ymm7 = ymm7[3,0,1,2]
	vpblendd	$192, %ymm4, %ymm5, %ymm8       # ymm8 = ymm5[0,1,2,3,4,5],ymm4[6,7]
	vpermq	$147, %ymm8, %ymm8              # ymm8 = ymm8[3,0,1,2]
	vpblendd	$192, %ymm5, %ymm2, %ymm9       # ymm9 = ymm2[0,1,2,3,4,5],ymm5[6,7]
	vpermq	$147, %ymm9, %ymm9              # ymm9 = ymm9[3,0,1,2]
	vpsllvq	%ymm0, %ymm3, %ymm3
	vpsllvq	%ymm0, %ymm4, %ymm4
	vpsllvq	%ymm0, %ymm5, %ymm5
	vpsllvq	%ymm0, %ymm2, %ymm10
	vpsrlvq	%ymm1, %ymm6, %ymm6
	vpor	%ymm6, %ymm3, %ymm3
	vpsrlvq	%ymm1, %ymm7, %ymm6
	vpor	%ymm6, %ymm4, %ymm4
	vpsrlvq	%ymm1, %ymm8, %ymm6
	vpor	%ymm6, %ymm5, %ymm5
	vpsrlvq	%ymm1, %ymm9, %ymm6
	vpor	%ymm6, %ymm10, %ymm6
	vmovdqu	%ymm3, (%rsi,%rdx,8)
	vmovdqu	%ymm4, 32(%rsi,%rdx,8)
	vmovdqu	%ymm5, 64(%rsi,%rdx,8)
	vmovdqu	%ymm6, 96(%rsi,%rdx,8)
	addq	$16, %rdx
	cmpq	%rdx, %rax
	jne	.LBB0_18
# %bb.19:                               # %middle.block67
	cmpq	%rax, %rcx
	je	.LBB0_22
# %bb.20:
	vextracti128	$1, %ymm2, %xmm0
	vpextrq	$1, %xmm0, %rdx
	.p2align	4
.LBB0_21:                               # %ushift_body
                                        # =>This Inner Loop Header: Depth=1
	shrxq	%r10, %rdx, %r8
	movq	(%r12,%rax,8), %rdx
	shlxq	%r15, %rdx, %rdi
	orq	%rdi, %r8
	movq	%r8, (%rsi,%rax,8)
	incq	%rax
	cmpq	%rax, %rcx
	jne	.LBB0_21
.LBB0_22:
	xorl	%edi, %edi
.LBB0_23:                               # %ushift_finish
	testq	%r15, %r15
	shrxq	%r10, -8(%r12,%rcx,8), %r11
	cmoveq	%r15, %r11
	movq	%r11, (%rsi,%rcx,8)
	movq	$0, 8(%rsi,%rcx,8)
	movq	%r10, %r8
	movq	-8(%r13,%r14,8), %r10
	movq	%r10, %rdx
	notq	%rdx
	movq	$-1, %rax
	#APP
	divq	%r10
	#NO_APP
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	cmpq	$1, %r14
	jne	.LBB0_27
# %bb.24:                               # %s_loop.preheader
	movq	%r8, %r13
	testb	%dil, %dil
	jne	.LBB0_62
# %bb.25:                               # %s_body.lr.ph
	movl	$1, %eax
	testb	$1, %cl
	jne	.LBB0_60
# %bb.26:
	movq	%rcx, %rax
	movq	-88(%rbp), %r12                 # 8-byte Reload
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	cmpq	$1, %rcx
	jne	.LBB0_61
	jmp	.LBB0_62
.LBB0_27:                               # %m_init
	movq	%r15, -104(%rbp)                # 8-byte Spill
	movq	%rbx, -112(%rbp)                # 8-byte Spill
	leaq	1(%rcx), %rdx
	testq	%r11, %r11
	cmoveq	%rcx, %rdx
	movq	%rdx, %rdi
	subq	%r14, %rdi
	js	.LBB0_51
# %bb.28:                               # %m_body.lr.ph
	movq	-48(%rbp), %r8                  # 8-byte Reload
	leaq	(%r8,%r14,8), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%r13,%r14,8), %r9
	movq	%r14, %rsi
	andq	$-2, %rsi
	movl	%r14d, %r11d
	andl	$3, %r11d
	movq	%r14, %rax
	andq	$-4, %rax
	movq	%rax, -120(%rbp)                # 8-byte Spill
	movq	%r10, %rax
	negq	%rax
	leaq	(,%r14,8), %rcx
	leaq	(%r8,%rdx,8), %rdx
	subq	%rcx, %rdx
	leaq	8(%rdx), %rbx
	movq	%rdx, -96(%rbp)                 # 8-byte Spill
	leaq	24(%rdx), %r15
	movq	%r14, -64(%rbp)                 # 8-byte Spill
	movq	%r9, -128(%rbp)                 # 8-byte Spill
	movq	%r11, -136(%rbp)                # 8-byte Spill
	jmp	.LBB0_31
	.p2align	4
.LBB0_29:                               # %ab_finish
                                        #   in Loop: Header=BB0_31 Depth=1
	decq	%rdx
	movq	-56(%rbp), %r8                  # 8-byte Reload
	addq	%rcx, (%r8,%rdi,8)
	movq	-64(%rbp), %r14                 # 8-byte Reload
	movq	-128(%rbp), %r9                 # 8-byte Reload
.LBB0_30:                               # %m_after
                                        #   in Loop: Header=BB0_31 Depth=1
	movq	-88(%rbp), %rcx                 # 8-byte Reload
	movq	%rdx, (%rcx,%rdi,8)
	addq	$-8, %rbx
	addq	$-8, %r15
	addq	$-8, -96(%rbp)                  # 8-byte Folded Spill
	testq	%rdi, %rdi
	leaq	-1(%rdi), %rdi
	jle	.LBB0_51
.LBB0_31:                               # %m_body
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_34 Depth 2
                                        #     Child Loop BB0_39 Depth 2
                                        #     Child Loop BB0_47 Depth 2
                                        #     Child Loop BB0_50 Depth 2
	movq	-56(%rbp), %rdx                 # 8-byte Reload
	movq	(%rdx,%rdi,8), %rcx
	movq	-16(%rdx,%rdi,8), %r8
	movq	-8(%rdx,%rdi,8), %r12
	cmpq	%r10, %rcx
	jae	.LBB0_33
# %bb.32:                               # %refine_check
                                        #   in Loop: Header=BB0_31 Depth=1
	movq	%rcx, %rdx
	mulxq	-80(%rbp), %r11, %r14           # 8-byte Folded Reload
	addq	%r12, %r11
	adcq	%rcx, %r14
	leaq	1(%r14), %rdx
	movq	%rdx, %rcx
	imulq	%r10, %rcx
	subq	%rcx, %r12
	cmpq	%r11, %r12
	cmovaq	%r14, %rdx
	movq	-64(%rbp), %r14                 # 8-byte Reload
	movl	$0, %ecx
	cmovaq	%r10, %rcx
	addq	%r12, %rcx
	cmpq	%r10, %rcx
	sbbq	$-1, %rdx
	cmpq	%r10, %rcx
	movl	$0, %r11d
	cmovaeq	%r10, %r11
	subq	%r11, %rcx
	movq	%rcx, %r12
	jmp	.LBB0_34
	.p2align	4
.LBB0_33:                               # %qhat_sat
                                        #   in Loop: Header=BB0_31 Depth=1
	movq	$-1, %rdx
	addq	%rcx, %r12
	jb	.LBB0_36
	.p2align	4
.LBB0_34:                               # %refine_loop
                                        #   Parent Loop BB0_31 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mulxq	%r9, %r11, %rcx
	cmpq	%r11, %r8
	movq	%r12, %r11
	sbbq	%rcx, %r11
	jae	.LBB0_36
# %bb.35:                               # %refine_step
                                        #   in Loop: Header=BB0_34 Depth=2
	decq	%rdx
	addq	%r10, %r12
	leaq	(%rax,%r12), %rcx
	cmpq	%rcx, %r12
	jae	.LBB0_34
.LBB0_36:                               # %do_submul
                                        #   in Loop: Header=BB0_31 Depth=1
	testq	%r14, %r14
	je	.LBB0_30
# %bb.37:                               # %sm_step.lr.ph
                                        #   in Loop: Header=BB0_31 Depth=1
	cmpq	$0, -144(%rbp)                  # 8-byte Folded Reload
	je	.LBB0_45
# %bb.38:                               # %sm_step.preheader
                                        #   in Loop: Header=BB0_31 Depth=1
	xorl	%r8d, %r8d
	xorl	%ecx, %ecx
	.p2align	4
.LBB0_39:                               # %sm_step
                                        #   Parent Loop BB0_31 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	mulxq	(%r13,%rcx,8), %r11, %r14
	addq	%r8, %r11
	adcq	$0, %r14
	subq	%r11, -8(%rbx,%rcx,8)
	adcq	$0, %r14
	mulxq	8(%r13,%rcx,8), %r11, %r8
	addq	%r14, %r11
	adcq	$0, %r8
	subq	%r11, (%rbx,%rcx,8)
	adcq	$0, %r8
	addq	$2, %rcx
	cmpq	%rcx, %rsi
	jne	.LBB0_39
# %bb.40:                               # %sm_finish.unr-lcssa
                                        #   in Loop: Header=BB0_31 Depth=1
	movq	-64(%rbp), %r14                 # 8-byte Reload
	testb	$1, %r14b
	je	.LBB0_42
.LBB0_41:                               # %sm_step.epil
                                        #   in Loop: Header=BB0_31 Depth=1
	movq	-48(%rbp), %r11                 # 8-byte Reload
	leaq	(%r11,%rdi,8), %r11
	mulxq	(%r13,%rcx,8), %r12, %r14
	addq	%r8, %r12
	adcq	$0, %r14
	subq	%r12, (%r11,%rcx,8)
	adcq	$0, %r14
	movq	%r14, %r8
	movq	-64(%rbp), %r14                 # 8-byte Reload
.LBB0_42:                               # %sm_finish
                                        #   in Loop: Header=BB0_31 Depth=1
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	subq	%r8, (%rcx,%rdi,8)
	jae	.LBB0_30
# %bb.43:                               # %ab_step.lr.ph
                                        #   in Loop: Header=BB0_31 Depth=1
	cmpq	$4, %r14
	jae	.LBB0_46
# %bb.44:                               #   in Loop: Header=BB0_31 Depth=1
	xorl	%r14d, %r14d
	xorl	%r8d, %r8d
	jmp	.LBB0_48
	.p2align	4
.LBB0_45:                               #   in Loop: Header=BB0_31 Depth=1
	xorl	%r8d, %r8d
	xorl	%ecx, %ecx
	movq	-64(%rbp), %r14                 # 8-byte Reload
	testb	$1, %r14b
	jne	.LBB0_41
	jmp	.LBB0_42
.LBB0_46:                               # %ab_step.preheader
                                        #   in Loop: Header=BB0_31 Depth=1
	xorl	%r14d, %r14d
	xorl	%r8d, %r8d
	movq	-120(%rbp), %r9                 # 8-byte Reload
	.p2align	4
.LBB0_47:                               # %ab_step
                                        #   Parent Loop BB0_31 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	xorl	%ecx, %ecx
	addq	(%r13,%r8,8), %r14
	setb	%cl
	addq	%r14, -24(%r15,%r8,8)
	adcq	8(%r13,%r8,8), %rcx
	setb	%r11b
	movzbl	%r11b, %r11d
	addq	%rcx, -16(%r15,%r8,8)
	adcq	16(%r13,%r8,8), %r11
	setb	%cl
	addq	%r11, -8(%r15,%r8,8)
	movzbl	%cl, %ecx
	adcq	24(%r13,%r8,8), %rcx
	setb	%r11b
	movzbl	%r11b, %r14d
	addq	%rcx, (%r15,%r8,8)
	adcq	$0, %r14
	addq	$4, %r8
	cmpq	%r8, %r9
	jne	.LBB0_47
.LBB0_48:                               # %ab_finish.unr-lcssa
                                        #   in Loop: Header=BB0_31 Depth=1
	movq	%r14, %rcx
	movq	-136(%rbp), %r9                 # 8-byte Reload
	testq	%r9, %r9
	je	.LBB0_29
# %bb.49:                               # %ab_step.epil.preheader
                                        #   in Loop: Header=BB0_31 Depth=1
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	leaq	(%rcx,%r8,8), %r12
	leaq	(,%r8,8), %r8
	addq	%r13, %r8
	xorl	%r11d, %r11d
	.p2align	4
.LBB0_50:                               # %ab_step.epil
                                        #   Parent Loop BB0_31 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	xorl	%ecx, %ecx
	addq	(%r8,%r11,8), %r14
	setb	%cl
	addq	%r14, (%r12,%r11,8)
	adcq	$0, %rcx
	incq	%r11
	movq	%rcx, %r14
	cmpq	%r11, %r9
	jne	.LBB0_50
	jmp	.LBB0_29
.LBB0_51:                               # %denorm_loop.preheader
	testq	%r14, %r14
	movq	-112(%rbp), %rbx                # 8-byte Reload
	movq	-104(%rbp), %r15                # 8-byte Reload
	movq	-72(%rbp), %r13                 # 8-byte Reload
	movq	-48(%rbp), %rsi                 # 8-byte Reload
	je	.LBB0_64
# %bb.52:                               # %d_step.lr.ph
	testq	%r15, %r15
	je	.LBB0_63
.LBB0_53:                               # %d_step.preheader
	movq	(%rsi), %rcx
	cmpq	$16, %r14
	jae	.LBB0_55
# %bb.54:
	xorl	%eax, %eax
	jmp	.LBB0_59
.LBB0_55:                               # %vector.ph74
	movq	%r14, %rax
	andq	$-16, %rax
	vmovq	%r15, %xmm0
	vpbroadcastq	%xmm0, %ymm0
	vmovq	%r13, %xmm1
	vpbroadcastq	%xmm1, %ymm1
	vmovq	%rcx, %xmm2
	vpbroadcastq	%xmm2, %ymm2
	xorl	%ecx, %ecx
	.p2align	4
.LBB0_56:                               # %vector.body81
                                        # =>This Inner Loop Header: Depth=1
	vmovdqu	8(%rsi,%rcx,8), %ymm3
	vmovdqu	40(%rsi,%rcx,8), %ymm4
	vmovdqu	72(%rsi,%rcx,8), %ymm5
	vpblendd	$192, %ymm2, %ymm3, %ymm6       # ymm6 = ymm3[0,1,2,3,4,5],ymm2[6,7]
	vmovdqu	104(%rsi,%rcx,8), %ymm2
	vpermq	$147, %ymm6, %ymm6              # ymm6 = ymm6[3,0,1,2]
	vpblendd	$192, %ymm3, %ymm4, %ymm7       # ymm7 = ymm4[0,1,2,3,4,5],ymm3[6,7]
	vpermq	$147, %ymm7, %ymm7              # ymm7 = ymm7[3,0,1,2]
	vpblendd	$192, %ymm4, %ymm5, %ymm8       # ymm8 = ymm5[0,1,2,3,4,5],ymm4[6,7]
	vpermq	$147, %ymm8, %ymm8              # ymm8 = ymm8[3,0,1,2]
	vpblendd	$192, %ymm5, %ymm2, %ymm9       # ymm9 = ymm2[0,1,2,3,4,5],ymm5[6,7]
	vpermq	$147, %ymm9, %ymm9              # ymm9 = ymm9[3,0,1,2]
	vpsrlvq	%ymm0, %ymm6, %ymm6
	vpsrlvq	%ymm0, %ymm7, %ymm7
	vpsrlvq	%ymm0, %ymm8, %ymm8
	vpsrlvq	%ymm0, %ymm9, %ymm9
	vpsllvq	%ymm1, %ymm3, %ymm3
	vpor	%ymm6, %ymm3, %ymm3
	vpsllvq	%ymm1, %ymm4, %ymm4
	vpor	%ymm7, %ymm4, %ymm4
	vpsllvq	%ymm1, %ymm5, %ymm5
	vpor	%ymm5, %ymm8, %ymm5
	vpsllvq	%ymm1, %ymm2, %ymm6
	vpor	%ymm6, %ymm9, %ymm6
	vmovdqu	%ymm3, (%rbx,%rcx,8)
	vmovdqu	%ymm4, 32(%rbx,%rcx,8)
	vmovdqu	%ymm5, 64(%rbx,%rcx,8)
	vmovdqu	%ymm6, 96(%rbx,%rcx,8)
	addq	$16, %rcx
	cmpq	%rcx, %rax
	jne	.LBB0_56
# %bb.57:                               # %middle.block89
	cmpq	%rax, %r14
	je	.LBB0_64
# %bb.58:
	vextracti128	$1, %ymm2, %xmm0
	vpextrq	$1, %xmm0, %rcx
	.p2align	4
.LBB0_59:                               # %d_step
                                        # =>This Inner Loop Header: Depth=1
	shrxq	%r15, %rcx, %rcx
	movq	8(%rsi,%rax,8), %rdx
	shlxq	%r13, %rdx, %rdi
	orq	%rcx, %rdi
	movq	%rdi, (%rbx,%rax,8)
	incq	%rax
	movq	%rdx, %rcx
	cmpq	%rax, %r14
	jne	.LBB0_59
	jmp	.LBB0_64
.LBB0_60:                               # %s_body.prol
	movq	-8(%rsi,%rcx,8), %rdi
	imulq	%r11, %rax
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	mulxq	%r11, %r11, %rsi
	addq	%rdi, %r11
	adcq	%rax, %rsi
	leaq	1(%rsi), %r8
	movq	%r8, %rax
	imulq	%r10, %rax
	subq	%rax, %rdi
	xorl	%r9d, %r9d
	cmpq	%r11, %rdi
	cmovaq	%rsi, %r8
	movl	$0, %r11d
	cmovaq	%r10, %r11
	addq	%rdi, %r11
	cmpq	%r10, %r11
	sbbq	$-1, %r8
	cmpq	%r10, %r11
	cmovaeq	%r10, %r9
	leaq	-1(%rcx), %rax
	subq	%r9, %r11
	movq	-48(%rbp), %rsi                 # 8-byte Reload
	movq	-88(%rbp), %r12                 # 8-byte Reload
	movq	%r8, -8(%r12,%rcx,8)
	cmpq	$1, %rcx
	je	.LBB0_62
	.p2align	4
.LBB0_61:                               # %s_body
                                        # =>This Inner Loop Header: Depth=1
	movq	-8(%rsi,%rax,8), %rdi
	movl	$1, %r8d
	mulxq	%r11, %rsi, %r9
	imulq	%r11, %r8
	addq	%rdi, %rsi
	adcq	%r8, %r9
	leaq	1(%r9), %rcx
	movq	%rcx, %r8
	imulq	%r10, %r8
	subq	%r8, %rdi
	cmpq	%rsi, %rdi
	cmovaq	%r9, %rcx
	movl	$0, %r8d
	cmovaq	%r10, %r8
	addq	%rdi, %r8
	cmpq	%r10, %r8
	sbbq	$-1, %rcx
	cmpq	%r10, %r8
	movl	$0, %esi
	cmovaeq	%r10, %rsi
	subq	%rsi, %r8
	movq	%rcx, -8(%r12,%rax,8)
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	-16(%rcx,%rax,8), %rdi
	movl	$1, %ecx
	imulq	%r8, %rcx
	mulxq	%r8, %rsi, %r8
	addq	%rdi, %rsi
	adcq	%rcx, %r8
	leaq	1(%r8), %r9
	movq	%r9, %rcx
	imulq	%r10, %rcx
	subq	%rcx, %rdi
	cmpq	%rsi, %rdi
	cmovaq	%r8, %r9
	movl	$0, %r11d
	cmovaq	%r10, %r11
	addq	%rdi, %r11
	cmpq	%r10, %r11
	sbbq	$-1, %r9
	cmpq	%r10, %r11
	movl	$0, %esi
	cmovaeq	%r10, %rsi
	subq	%rsi, %r11
	movq	%r9, -16(%r12,%rax,8)
	movq	-48(%rbp), %rsi                 # 8-byte Reload
	addq	$-2, %rax
	testq	%rax, %rax
	jne	.LBB0_61
.LBB0_62:                               # %denorm_loop.preheader.thread
	movq	%r11, (%rsi)
	movq	$0, 8(%rsi)
	testq	%r15, %r15
	jne	.LBB0_53
.LBB0_63:                               # %d_step.us.preheader
	shlq	$3, %r14
	movq	%rbx, %rdi
	movq	%r14, %rdx
	vzeroupper
	callq	memcpy@PLT
.LBB0_64:                               # %exit
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	retq
.Lfunc_end0:
	.size	zfactor_mg_divrem, .Lfunc_end0-zfactor_mg_divrem
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
