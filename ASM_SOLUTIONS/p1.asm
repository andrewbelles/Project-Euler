section .note.GNU-stack noalloc noexec nowrite progbits

section .rodata
  count db "Sum: %s", 10, 0
  iform db "%d", 0

section .text
  extern sprintf, printf
  global main

main: 


  ; loop from 0 to 1000. take the modulus of n w 3 and then 5 and add if either is true. mod 3 will be more common so do    second

  ; integer division places the result in rax:rdx. rdx is the remainder 

  xor rcx, rcx 
  xor rbx, rbx
  mov r8, 5
  mov r9, 3

loop: 
  
  inc rcx           ; nth value to be divided
  cmp rcx, 1000
  jge results

  mov rax, rcx
  xor rdx, rdx 
  idiv r8
  test rdx, rdx
  jz sum
  
  mov rax, rcx
  xor rdx, rdx 
  idiv r9
  test rdx, rdx 
  jnz loop

sum: 
  add rbx, rcx  
  jmp loop

results: 

  ; print results 
  sub rsp, 24
  ; sum is stored in rbx
  mov rdx, rbx 
  lea rdi, [rsp]
  lea rsi, [iform]
  mov rax, 1 
  call sprintf 

  lea rdi, [count]
  lea rsi, [rsp] 
  xor rax, rax 
  call printf 

  add rsp, 24
  mov rax, 60
  xor rdi, rdi
  syscall
