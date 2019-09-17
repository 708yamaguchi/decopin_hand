(send *fetch* :angle-vector #f(48.096 38.5848 -5.11674 -109.758 107.615 -196.853 72.4581 107.954 0.305897 15.7652))
(send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
(send *ri* :wait-interpolation)

(send *ri* :angle-vector (send *fetch* :rarm :move-end-pos #f(0 0 -50) :world :use-torso nil) 3000)
(send *ri* :wait-interpolation)

(send *ri* :angle-vector (send *fetch* :rarm :move-end-pos #f(0 0 50) :world :use-torso nil) 3000)
(send *ri* :wait-interpolation)
