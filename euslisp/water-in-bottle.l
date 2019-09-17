;; /hit_volume_threshold: 0.09

;; shake bottle
(do-until-key
 (send *ri* :angle-vector (send *fetch* :angle-vector #f(48.096 25.0667 63.4532 -149.067 114.318 -175.933 48.3041 40.0 0.305897 15.7652)) 3000)
 (send *ri* :wait-interpolation)

 (send *ri* :angle-vector (send *fetch* :angle-vector #f(48.096 17.5447 56.4598 -150.959 116.736 -171.816 59.0524 81.093 0.305897 15.7652)) 3000)
 (send *ri* :wait-interpolation))
