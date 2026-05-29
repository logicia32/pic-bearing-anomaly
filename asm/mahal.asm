;;;==========================================================================
;;;  mahal.asm  --  PIC16F1503 上で 2D マハラノビス距離の異常検知
;;;
;;;  入力: in_x_q (Q14 unsigned 16-bit), in_y_q (同)
;;;  出力: W = 0 healthy / W = 1 anomaly
;;;
;;;  d² = c·dx² − 2b·dx·dy + a·dy²  (除算ゼロ / sqrt 不要)
;;;
;;;  モデル定数 (NASA IMS Bearing 3, ε=2e-5 正則化)
;;;     mu_x_q = 2276 (0x08E4),  mu_y_q = 2320 (0x0910)
;;;     k_a    = 26188 (0x664C)
;;;     k_b    = 23062 (0x5A16)
;;;     k_c    = 26081 (0x65E1)
;;;     T_q    = 11898 (0x2E7A)
;;;
;;;  PIC16F1503: enhanced mid-range, no hardware multiplier。
;;;==========================================================================

        PROCESSOR 16F1503
        #include <p16f1503.inc>

        __CONFIG _CONFIG1, _FOSC_INTOSC & _WDTE_OFF & _PWRTE_ON & _MCLRE_ON & _CP_OFF & _BOREN_ON & _CLKOUTEN_OFF
        __CONFIG _CONFIG2, _WRT_OFF & _STVREN_ON & _BORV_LO & _LPBOR_OFF & _LVP_OFF

;;;-------------------- RAM 配置 (Bank 0: 0x20-0x6F) ----
        CBLOCK  0x20
in_x_lo
in_x_hi
in_y_lo
in_y_hi
        ENDC

        CBLOCK  0x30
dx_lo
dx_hi
dy_lo
dy_hi
acc0
acc1
acc2
acc3
mA_lo
mA_hi
mB_lo
mB_hi
prod0
prod1
prod2
prod3
        ENDC

        CBLOCK  0x50
tmp_a
cnt
sign
mul_b
pp_lo
pp_hi
result          ; 0=healthy / 1=anomaly (main 終了後の保存先)
        ENDC

;;;-------------------- Reset vector ---------------------
        ORG     0x0000
        goto    main

;;;==========================================================================
;;;  mul8x8u : 8x8 → 16-bit 符号無し乗算
;;;     入力:  W = A,  mul_b = B
;;;     出力:  pp_hi:pp_lo
;;;==========================================================================
        ORG     0x0004
mul8x8u:
        movwf   tmp_a
        clrf    pp_lo
        clrf    pp_hi
        movlw   .8
        movwf   cnt
m8_loop:
        rrf     tmp_a, f
        btfsc   STATUS, C
        goto    m8_add
        bcf     STATUS, C
        goto    m8_shift
m8_add:
        movf    mul_b, w
        addwf   pp_hi, f
m8_shift:
        rrf     pp_hi, f
        rrf     pp_lo, f
        decfsz  cnt, f
        goto    m8_loop
        return

;;;==========================================================================
;;;  mul16x16u : 16x16 → 32-bit 符号無し乗算
;;;     入力:  mA_hi:mA_lo, mB_hi:mB_lo (16-bit unsigned)
;;;     出力:  prod3:prod2:prod1:prod0 (32-bit unsigned)
;;;==========================================================================
mul16x16u:
        clrf    prod0
        clrf    prod1
        clrf    prod2
        clrf    prod3

        ; A_lo * B_lo  →  prod[0..1]
        movf    mB_lo, w
        movwf   mul_b
        movf    mA_lo, w
        call    mul8x8u
        movf    pp_lo, w
        movwf   prod0
        movf    pp_hi, w
        movwf   prod1

        ; A_lo * B_hi  →  prod[1..2]
        movf    mB_hi, w
        movwf   mul_b
        movf    mA_lo, w
        call    mul8x8u
        movf    pp_lo, w
        addwf   prod1, f
        movlw   0
        addwfc  prod2, f
        addwfc  prod3, f
        movf    pp_hi, w
        addwf   prod2, f
        movlw   0
        addwfc  prod3, f

        ; A_hi * B_lo  →  prod[1..2]
        movf    mB_lo, w
        movwf   mul_b
        movf    mA_hi, w
        call    mul8x8u
        movf    pp_lo, w
        addwf   prod1, f
        movlw   0
        addwfc  prod2, f
        addwfc  prod3, f
        movf    pp_hi, w
        addwf   prod2, f
        movlw   0
        addwfc  prod3, f

        ; A_hi * B_hi  →  prod[2..3]
        movf    mB_hi, w
        movwf   mul_b
        movf    mA_hi, w
        call    mul8x8u
        movf    pp_lo, w
        addwf   prod2, f
        movlw   0
        addwfc  prod3, f
        movf    pp_hi, w
        addwf   prod3, f

        return

;;;==========================================================================
;;;  neg16_A / neg16_B : 16-bit 2's complement negate (in-place)
;;;==========================================================================
neg16_A:
        comf    mA_lo, f
        comf    mA_hi, f
        incf    mA_lo, f
        btfsc   STATUS, Z
        incf    mA_hi, f
        return

neg16_B:
        comf    mB_lo, f
        comf    mB_hi, f
        incf    mB_lo, f
        btfsc   STATUS, Z
        incf    mB_hi, f
        return

;;;==========================================================================
;;;  neg32 : 32-bit prod negate (in-place)
;;;==========================================================================
neg32:
        comf    prod0, f
        comf    prod1, f
        comf    prod2, f
        comf    prod3, f
        incf    prod0, f
        btfsc   STATUS, Z
        incf    prod1, f
        btfsc   STATUS, Z
        incf    prod2, f
        btfsc   STATUS, Z
        incf    prod3, f
        return

;;;==========================================================================
;;;  mul16x16s : 16x16 → 32-bit 符号付き乗算
;;;     |A|×|B| を unsigned で計算し、結果符号を復元.
;;;==========================================================================
mul16x16s:
        clrf    sign
        btfsc   mA_hi, 7
        bsf     sign, 0
        btfsc   mB_hi, 7
        incf    sign, f         ; XOR with bit 0
        btfsc   mA_hi, 7
        call    neg16_A
        btfsc   mB_hi, 7
        call    neg16_B
        call    mul16x16u
        btfss   sign, 0
        return
        call    neg32
        return

;;;==========================================================================
;;;  acc_add32 / acc_sub32 : 32-bit signed acc ± prod
;;;==========================================================================
acc_add32:
        movf    prod0, w
        addwf   acc0, f
        movf    prod1, w
        addwfc  acc1, f
        movf    prod2, w
        addwfc  acc2, f
        movf    prod3, w
        addwfc  acc3, f
        return

acc_sub32:
        movf    prod0, w
        subwf   acc0, f
        movf    prod1, w
        subwfb  acc1, f
        movf    prod2, w
        subwfb  acc2, f
        movf    prod3, w
        subwfb  acc3, f
        return

;;;==========================================================================
;;;  mah_evaluate : 単発評価
;;;     入力: in_x_*, in_y_*
;;;     出力: W = 0 (healthy) / W = 1 (anomaly)
;;;==========================================================================
mah_evaluate:
        ; dx = in_x - 2276
        movlw   0xE4
        subwf   in_x_lo, w
        movwf   dx_lo
        movlw   0x08
        subwfb  in_x_hi, w
        movwf   dx_hi

        ; dy = in_y - 2320
        movlw   0x10
        subwf   in_y_lo, w
        movwf   dy_lo
        movlw   0x09
        subwfb  in_y_hi, w
        movwf   dy_hi

        ; acc = 0
        clrf    acc0
        clrf    acc1
        clrf    acc2
        clrf    acc3

        ;;; ---- c * dx²  →  acc += result ----
        ; prod = dx * dx (signed)
        movf    dx_lo, w
        movwf   mA_lo
        movf    dx_hi, w
        movwf   mA_hi
        movf    dx_lo, w
        movwf   mB_lo
        movf    dx_hi, w
        movwf   mB_hi
        call    mul16x16s

        ; prod = (prod >> 16) * k_c  →  signed 16 × signed 16 → signed 32
        movf    prod2, w
        movwf   mB_lo
        movf    prod3, w
        movwf   mB_hi
        movlw   0xE1                ; k_c = 0x65E1
        movwf   mA_lo
        movlw   0x65
        movwf   mA_hi
        call    mul16x16s
        call    acc_add32

        ;;; ---- a * dy²  →  acc += result ----
        movf    dy_lo, w
        movwf   mA_lo
        movf    dy_hi, w
        movwf   mA_hi
        movf    dy_lo, w
        movwf   mB_lo
        movf    dy_hi, w
        movwf   mB_hi
        call    mul16x16s
        movf    prod2, w
        movwf   mB_lo
        movf    prod3, w
        movwf   mB_hi
        movlw   0x4C                ; k_a = 0x664C
        movwf   mA_lo
        movlw   0x66
        movwf   mA_hi
        call    mul16x16s
        call    acc_add32

        ;;; ---- 2 * b * dx * dy  →  acc -= result ----
        movf    dx_lo, w
        movwf   mA_lo
        movf    dx_hi, w
        movwf   mA_hi
        movf    dy_lo, w
        movwf   mB_lo
        movf    dy_hi, w
        movwf   mB_hi
        call    mul16x16s

        ; prod *= 2  (32-bit left shift 1)
        bcf     STATUS, C
        rlf     prod0, f
        rlf     prod1, f
        rlf     prod2, f
        rlf     prod3, f

        ; (prod >> 16) * k_b
        movf    prod2, w
        movwf   mB_lo
        movf    prod3, w
        movwf   mB_hi
        movlw   0x16                ; k_b = 0x5A16
        movwf   mA_lo
        movlw   0x5A
        movwf   mA_hi
        call    mul16x16s
        call    acc_sub32

        ;;; ---- しきい値比較: acc > T_q ?  (T_q = 0x2E7A) ----
        ; acc が負なら確実に healthy
        btfsc   acc3, 7
        goto    healthy
        ; acc[3] != 0 → 確実に T 超え (acc は 32-bit 範囲、T は 16-bit)
        movf    acc3, w
        btfss   STATUS, Z
        goto    anomaly
        movf    acc2, w
        btfss   STATUS, Z
        goto    anomaly
        ; acc[3:2]=0; acc[1:0] と T_q (0x2E7A) を 16-bit 比較
        movlw   0x2E
        subwf   acc1, w             ; W = acc1 - 0x2E
        btfss   STATUS, Z
        goto    cmp_done
        movlw   0x7A
        subwf   acc0, w             ; (acc1 == 0x2E) なら下位も比較
cmp_done:
        btfsc   STATUS, C           ; C=1 → acc[1:0] >= T_q
        goto    anomaly

healthy:
        retlw   0

anomaly:
        retlw   1

;;;==========================================================================
;;;  main : デモ
;;;     in_x = 9727 (=0.5936 g @ Q14 = 末期 anomaly)
;;;     in_y = 8127 (=0.4960 g @ Q14)
;;;==========================================================================
main:
        movlw   0xFF
        movwf   in_x_lo
        movlw   0x25
        movwf   in_x_hi
        movlw   0xBF
        movwf   in_y_lo
        movlw   0x1F
        movwf   in_y_hi

        call    mah_evaluate
        movwf   result          ; result = W (0=healthy / 1=anomaly)
loop:
        goto    loop

        END
