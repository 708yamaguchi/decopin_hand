#!/usr/bin/env roseus

;; Before using this script, please create euslisp model
;; based on README.md

(load "package://euscollada/src/euscollada-robot.l")
(load "package://decopin_hand/euslisp/decopin_hand_utils.l")

(setq *robot* (decopin_hand))

(objects (list *robot*))

(dotimes (i 360)
  (send *robot* :angle-vector (float-vector i i i i))
  (format t "angle-vector: #f(~A ~A ~A ~A)~%" i i i i)
  (send *irtviewer* :redraw)
  (unix:usleep (* 10 1000)))
