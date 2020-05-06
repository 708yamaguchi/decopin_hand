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


;; Before demo, make sure that 3 fingers are opened and wire is not tied on the left finger
(defun main ()
  (ros::subscribe "/action_classifier/output"
                  jsk_recognition_msgs::ClassificationResult #'cb)
  (ros::rate 100)
  (setq *classify-result* nil)
  (print "Please start demo")
  ;; close finger after touch_nail
  (do-until-key
   (ros::spin-once)
   (when *classify-result*
     (send *ri* :angle-vector
           (send *robot* :angle-vector
                 (v+ (send *robot* :angle-vector) #f(0 0 0 -140)))
           3000)
     (send *ri* :wait-interpolation)
     (return)))
  (unix:sleep 3)
  (print "Please start fixing finger")
  (setq *classify-result* nil)
  ;; fix finger by human and open finger after touch_nail
  (do-until-key
   (ros::spin-once)
   (when *classify-result*
     (send *ri* :angle-vector
           (send *robot* :angle-vector
                 (v+ (send *robot* :angle-vector) #f(0 0 0 140)))
           3000)
     (send *ri* :wait-interpolation)
     (return)))
  (print "Finish fixing finger")
  ;; open and close gripper to show the fix is successfully completed
  (dotimes (i 2)
    (print "Close gripper")
    (send *ri* :angle-vector
          (send *robot* :angle-vector
                (v+ (send *robot* :angle-vector) #f(55 0 55 -55)))
          1000)
    (send *ri* :wait-interpolation)
    (print "Open gripper")
    (send *ri* :angle-vector
          (send *robot* :angle-vector
                (v+ (send *robot* :angle-vector) #f(-55 0 -55 55)))
          1000)
    (send *ri* :wait-interpolation)
    )
  )
