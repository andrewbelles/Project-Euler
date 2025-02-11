section .note.GNU-stack noalloc noexec nowrite progbits

section .rodata
  count db "Start: %d", 10, 0
  chain db "Chain: %d", 10, 0

section .text
  extern sprintf, printf
  global main

main:

  ; max cached data is 1e5 
  mov rsi, 0x186A0
  imul rsi, 8
  add rsi, 8
  
  ; allocate cache
  mov rax, 9
  xor rdi, rdi
  mov rdx, 3
  mov r10, 34
  mov r8, -1 
  xor r9, r9
  syscall
  ; assume success
  
  test rax, rax
  js error

  ; move array to r15 
  mov r15, rax 

  ; set cache to zero
  xor rcx, rcx 
.zero:

  ; increment point and move 0 to location
  mov rsi, rcx 
  imul rsi, 8
  mov qword [r15 + rsi], 0x0
  inc rcx

  cmp rcx, 0x186A0
  jle .zero

  ; initialize n = 1 to 1
  mov rsi, 8
  mov qword [r15 + rsi], 0x1
  
  ; init value counters
  xor rcx, rcx 
  xor r14, r14 
  xor r13, r13 
.loop:
  
  cmp rcx, 0x186A0
  jg .end

  inc rcx

  ; collatz call for rcx 
  push rcx
  call collatz
  pop rcx

  cmp rax, r14
  jle .loop

  mov r14, rax 
  mov r13, rcx

  jmp .loop

.end:

  push rcx
  sub rsp, 8

  ; print chain length
  mov rax, r14
  lea rdi, [chain]
  mov rsi, rax 
  xor rax, rax
  call printf 

  add rsp, 8
  pop rcx

  ; print value that produced chain (answer)
  mov rcx, r13
  lea rdi, [count]
  mov rsi, rcx
  xor rax, rax 
  call printf

  mov rax, 60
  xor rdi, rdi
  syscall

; rax holds the sequence/chain count  
; rcx holds the n to be calculated 
collatz:

  mov r10, rcx 
  xor r8, r8
  xor rax, rax 
.loop: 

  ; compare rcx to 1 (done)
  cmp rcx, 1
  je .result

  ; compare rcx to max data
  cmp rcx, 0x186A0
  jg .nocache

  push r8 
  mov r9, rcx 
  imul r9, 8
  mov r8, [r15 + r9]
  cmp r8, 0
  jz .nocache

  pop r8
  ; pull cached value 
  add rax, r8

  ret

  ; no available cached value for n
  .nocache:
    pop r8    
    ; bitwise and 
    test rcx, 1 
    jz .evenc
    jnz .oddc

    .evenc:
    push rcx   
    push rax

    mov rax, rcx
    xor rdx, rdx
    mov rcx, 2
    idiv rcx

    mov rcx, rax 
    pop rax  
    pop rcx

    .oddc:

    imul rcx, 3
    add rcx, 1

    inc r8

    jmp .loop

.result:
  
  mov r11, r10 
  imul r11, 8
  mov [r15 + r11], r8

  mov rax, r8

  ret



error: 

  mov rax, 60
  xor rdi, rdi
  syscall
