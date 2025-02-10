section .note.GNU-stack noalloc noexec nowrite progbits

section .rodata
  max db "Max: %f", 10, 0
  index db "Index: %d", 10, 0

section .text
  extern sprintf, printf
  global main

main:

  mov rax, 0xF4240 
  ; put the prime factorial in rcx 
  call primefac 

  push rcx
  sub rsp, 24

  ; collect totient of rcx in rax 
  call totient

  add rsp, 24
  pop rcx

  ; store the ratio in xmm0 
  cvtsi2sd xmm0, rcx 
  cvtsi2sd xmm1, rax
  divsd xmm0, xmm1 
  
  sub rsp, 16
  movsd [rsp + 8], xmm0
  ; Index print
  lea rdi, [index]
  mov rsi, rcx 
  xor rax, rax 
  call printf 

  ; Max Ratio print
  movsd xmm0, [rsp + 8]

  lea rdi, [max]
  mov al, 1 
  call printf
  add rsp, 16

  ; sysexit
  mov rax, 60
  xor rdi, rdi 
  syscall

; value passed in rcx
; n passed back in rax 
totient:
  push rbp 
  mov rbp, rsp
  push rbx 

  mov rbx, rcx

  cmp rcx, 0
  jle .base 

  mov r8, 1
  xor rcx, rcx 

  .loop: 
    
    cmp r8, rbx 
    jge .res 
  
    ; reset copy value to full value for each while loop
    mov r9, rbx
    mov r10, r8
    
  .while:
     
    test r9, r9 
    jz .test

    mov r11, r9               ; tmp = copy
    mov rax, r10              ; rax holds div
    xor rdx, rdx 
    div r9                    ; rdx hold rem
    
    mov r9, rdx               ; copy = div % copy
    mov r10, r11              ; div = tmp

    jmp .while

  .test:

    cmp r10, 1
    jne .next 
    inc rcx                   ; if div == 1 count++

  .next: 

    inc r8
    jmp .loop

  .res:
    mov rax, rcx
    mov rsp, rbp
    pop rbx
    pop rbp
    ret

  .base: 
    xor rax, rax              ; input value is <= 0 
    mov rsp, rbp 
    pop rbx
    pop rbp
    ret 

; takes in count through rax and returns prime factorial in rcx 
primefac:
  
  mov rcx, 1 
  xor r8, r8

.loop: 

    inc r8

    push rcx 
    push rax

    call next_prime
    
    mov r8, rax

    pop rax 
    pop rcx 

    mov r9, rcx
    imul r9, r8
    
    cmp r9, rax
    jge .res

    jmp .loop

.res:
    ; return the prime factorial in rcx 
    ret

next_prime:

  .loop:

    mov rcx, r8
    push r8
    sub rsp, 8

    call check_prime 
    
    add rsp, 8
    pop r8

    cmp rax, 1 
    je .prime 

    inc r8
    jmp .loop

  .prime: 
    mov rax, r8
    ret                     ; next prime value store in r8

check_prime:
  ; check if less than or equal to 3 
  cmp rcx, 2
  jl .false 

  ; 2 and 3 are prime
  cmp rcx, 3
  jle .true

  ; check if 2 is factor
  mov rax, rcx 
  mov r8, 2
  xor rdx, rdx 
  idiv r8 
  test rdx, rdx 
  jz .false

  ; check if 3 is factor 
  mov rax, rcx 
  mov r8, 3
  xor rdx, rdx 
  idiv r8 
  test rdx, rdx 
  jz .false 

  ; skipping over multiples of 2 and 3 check divisors up to sqrt(n)
  mov r9, 5
  .prime_loop: 

    ; test if r9 is a factor 
    mov rax, rcx
    div r9
    test rdx, rdx 
    jz .false

    ; test if r9 + 2 is a factor 
    mov rax, rcx 
    mov r8, 2 
    add r8, r9
    div r8 
    test rdx, rdx
    jz .false 

    ; if neither were factors and r9^2 >= n then its prime 
    mov r10, r9
    imul r10, r10
    cmp r10, rcx 
    jge .true  

    ; if not done iterate by 6 and reloop
    add r9, 6
    jmp .prime_loop 

  .true:
    mov rax, 1 
    ret

  .false: 
    mov rax, 0
    ret
