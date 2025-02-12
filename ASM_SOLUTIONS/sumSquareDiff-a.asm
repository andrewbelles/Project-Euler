section .note.GNU-stack noalloc noexec nowrite progbits

section .rodata
  count db "Difference: %d", 10, 0

section .text
  extern sprintf, printf
  global main

main: 

  xor rcx, rcx  ; sum of squares 
  xor rax, rax  ; square sum
  mov rdx, 1    ; i
.loop: 
  
  add rax, rdx 

  mov r8, rdx
  imul r8, r8
  add rcx, r8

  inc rdx
  cmp rdx, 0x64
  jle .loop 

  imul rax, rax 

  sub rax, rcx 

  lea rdi, [count]
  mov rsi, rax 
  xor rax, rax 
  call printf 

  mov rax, 60
  xor rdi, rdi
  syscall


  

