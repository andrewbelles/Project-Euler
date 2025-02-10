section .note.GNU-stack noalloc noexec nowrite progbits

section .rodata
  count db "Highest Prime Factor: %d", 10, 0

section .text
  extern printf
  global main

main:

  mov rbx, qword 0x8BE589EAC7

  mov rcx, 3
  xor rdx, rdx          ; holds highest prime factor

loop: 

  push rcx

  call check_prime

  pop rcx

  cmp rax, 0 
  je .skip 

  mov rdx, rcx

  .skip: 
    inc rcx 
    cmp rcx, rbx
    jge end

  jmp loop

end: 
  
  sub rsp, 24

  lea rdi, [count]
  mov rsi, rdx 
  xor rax, rax 
  call printf 

  add rsp, 24
  mov rax, 60
  xor rdi, rdi
  syscall


; prime "sieve" asm implementation
; rcx holds n   
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
