#!/usr/bin/env roseus

(require "package://decopin_hand/euslisp/decopin_hand_interface.l")
(ros::load-ros-manifest "jsk_recognition_msgs")

(decopin-hand-init)
(setq *classify-result* nil)

(defun cb (msg)
  (setq a msg)
  (setq *classify-result* (send msg :classifier)))


(defun main ()
  (ros::subscribe "/action_classifier/output"
                  jsk_recognition_msgs::ClassificationResult #'cb)
  (ros::rate 100)
  (do-until-key
   (ros::spin-once)
   (when (equal *classify-result* "touch_nail\n")
     (print "touch nail !!")
     (send *ri* :angle-vector
           (v+ (send *robot* :angle-vector) #f(-30 0 0 0)))
     (send *ri* :wait-interpolation)
     (return-from main)))
  )
