#!/usr/bin/env roseus

(require "package://decopin_hand/euslisp/decopin_hand_interface.l")
(ros::load-ros-manifest "jsk_recognition_msgs")

(decopin-hand-init)
(send *robot* :angle-vector (send *ri* :state :potentio-vector))

(defun cb (msg)
  (setq a msg)
  (when (equal (car (send msg :label_names)) (format nil "touch_nail~%"))
    (print "touch nail !!")
    (setq *classify-result* t))
  )


(defun main ()
  (ros::subscribe "/action_classifier/output"
                  jsk_recognition_msgs::ClassificationResult #'cb)
  (ros::rate 100)
  (setq *classify-result* nil)
  (print "Please start demo")
  (do-until-key
   (ros::spin-once)
   (when *classify-result*
     (send *ri* :angle-vector
           (send *robot* :angle-vector
                 (v+ (send *robot* :angle-vector) #f(-60 0 0 0)))
           3000)
     (send *ri* :wait-interpolation)
     (return-from main)))
  )
