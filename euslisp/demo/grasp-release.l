#!/usr/bin/env roseus

;; Mainly copied from soldering.l

(require "package://decopin_hand/euslisp/decopin_hand_interface.l")
(ros::load-ros-manifest "jsk_recognition_msgs")

(decopin-hand-init)
(send *robot* :angle-vector (send *ri* :state :potentio-vector))

;; states are :grasp and :release
(setq *state* :release)

(defun cb (msg)
  (setq a msg)
  (if (equal (car (send msg :label_names)) (format nil "touch_door~%"))
    (progn
      (print "touch door !!")
      (setq *classify-result* t))
    (setq *classify-result* nil))
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
     (cond
      ((equal *state* :release)
       (send *robot* :angle-vector
             (v+ (send *robot* :angle-vector) #f(80 0 0 -80)))
       (setq *state* :grasp))
      ((equal *state* :grasp)
       (send *robot* :angle-vector
             (v+ (send *robot* :angle-vector) #f(-80 0 0 80)))
       (setq *state* :release)))
     (send *ri* :angle-vector (send *robot* :angle-vector) 2000)
     (format t "Moving...~%")
     (unix:sleep 10)
     (format t "*state* is ~A, wait for input...~%" *state*)
     (send *ri* :wait-interpolation)))
  (return-from main)
  )
