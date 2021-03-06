;; /hit_volume_threshold: 0.09
(ros::load-ros-manifest "sound_classification")

(setq max-sound 0)
(defun cb (msg)
  (format t "volume:~%~A~%" (send msg :volume))
  (format t "max-sound:~%~A~%" max-sound)
  (when (> (send msg :volume) 0.05)
    ;; store max sound
    (if (> (send msg :volume) max-sound)
      (setq max-sound (send msg :volume))))
  )
(ros::subscribe "/microphone/volume" sound_classification::Volume #'cb)

;; part1
(setq part-1 (send *fetch* :angle-vector #f(48.096 15.0535 -30.9713 -94.04 32.7146 -284.882 117.751 247.523 0.305897 15.7652)))
;; part2


;; send commmands to *ri*
;; (send *ri* :angle-vector (send *fetch* :angle-vector #f(48.096 39.7656 7.20718 -98.7791 111.423 -180.14 71.1149 101.385 0.305897 15.7652)) 3000) ;; initial-pose
;; (send *ri* :wait-interpolation)
;; move hand
(send *ri* :angle-vector (send *fetch* :rarm :move-end-pos #f(0 0 -50) :world :use-torso nil) 3000)

;; hear max sound
(while
  (unix:usleep 1000)
  (ros::spin-once))
