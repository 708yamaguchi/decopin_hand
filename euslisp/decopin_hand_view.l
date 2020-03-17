(load "package://euscollada/src/euscollada-robot.l")
(load "package://decopin_hand/model/decopin_hand.l")

(setq *robot* (decopin_hand))

(objects (list *robot*))

(dotimes (i 360)
  (send *robot* :angle-vector (float-vector i i i i))
  (send *irtviewer* :redraw)
  (unix:usleep (* 10 1000)))
