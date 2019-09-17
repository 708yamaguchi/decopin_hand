;; /hit_volume_threshold: 0.09

;; fetch
(send *fetch* :angle-vector #f(48.096 39.3174 0.969118 -103.429 110.474 -188.126 71.9606 103.592 0.305897 15.7652))
(send *ri* :angle-vector (send *fetch* :angle-vector) 10000)

;; decopin-hand
(send *robot* :angle-vector (v+ (send *robot* :reset-pose) #f(0 110 -110 110)))
(send *ri* :angle-vector (send *robot* :angle-vector))

(send *robot* :angle-vector (v+ (send *robot* :angle-vector) #f(0 50 0 0)))
(send *ri* :angle-vector (send *robot* :angle-vector) 3000)
