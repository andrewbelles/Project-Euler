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
  sub rsp, 8

  ; collect totient of rcx in rax 
  call totient

  add rsp, 8
  pop rcx

  ; store the ratio in xmm0 
  cvtsi2sd xmm0, rcx 
  cvtsi2sd xmm1, rax
  divsd xmm0, xmm1 
  
  sub rsp, 24
  movsd [rsp], xmm0

  lea rdi, [index]
  mov rsi, rcx 
  xor rax, rax 
  call printf 

  movsd xmm0, [rsp]
  add rsp, 24

  lea rdi, [max]
  xor rax, rax 
  call printf

  mov rax, 60
  xor rdi, rdi 
  syscall

; value passed in rcx
; n passed back in rax 
totient: 

  cmp rcx, 0
  jle .base 

  mov r8, 1
  mov r9, rcx 
  xor rcx, rcx 

  .loop: 
    mov r10, r8
    
    .while:
      
      mov r11, r9 
      mov rax, r10 
      idiv r11 
      mov r10, r11 
      
      cmp rdx, 1
      je .count 

      test rdx, rdx 
      jnz .while 

    .count:
    
      inc rcx

    inc r8
    cmp r8, r9
    je .res

  .res:
    mov rax, rcx
    ret

  .base: 
    xor rcx, rcx 
    ret 

; takes in count through rax and returns prime factorial in rcx 
primefac:
  
  mov rcx, 1 
  xor r8, r8

.loop: 

    push rcx 
    push rax

    inc r8
    call next_prime

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
    call check_prime 

    cmp rax, 1 
    je .prime 

    inc r8
    jmp .loop

  .prime: 

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
