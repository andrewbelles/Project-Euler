section .note.GNU-stack noalloc noexec nowrite progbits

section .rodata
  count db "Sum: %s", 10, 0
  iform db "%d", 0

section .text
  extern sprintf, printf
  global main

main: 

  ; init accumulators 
  mov rdx, 2                  ; loop will miss 2 since add comes before check 
  mov rcx, 1 
  mov rbx, 2 
loop: 
  
  ; a = a + b 
  add rcx, rbx
  
  ; swap two values with copying 
  xor rcx, rbx 
  xor rbx, rcx 
  xor rcx, rbx                ; b = a, a = b

  ; check if value is even and add if so 
  mov rax, rbx
  and rax, 1
  cmp rax, 0 
  jne odd 

  add rdx, rbx 

odd: 

  ; compare with 4 million
  cmp rbx, qword 0x3D0900
  jle loop

  ; print result 
  sub rsp, 24 

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
