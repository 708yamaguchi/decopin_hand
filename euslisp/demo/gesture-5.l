#!/usr/bin/env roseus

;; Mainly copied from soldering.l

(require "package://decopin_hand/euslisp/decopin_hand_utils.l")
(ros::load-ros-manifest "jsk_recognition_msgs")

(decopin-hand-init)
(send *robot* :angle-vector (send *ri* :state :potentio-vector))
;; States are :release, :grasp-1-finger, :grasp-2-finger
;; Make sure that the hand is opening at first
(setq *state* :release)

(defun main ()
  (print "Please start demo")
  (ros::rate 30)
  (do-until-key
   (ros::spin-once)
   (let ((av-list nil) (loop-continue t))
     ;; Set angle-vector and check
     (cond
      ;; open 1 finger
      ((equal *continuous-class* "rub_finger") ;; high_long
       (if (eq *state* :grasp-1-finger)
         (progn
           (setq av-list (list (v+ (send *robot* :angle-vector) #f(0 0 0 140))))
           (setq *state* :release))
         (setq loop-continue nil)))
      ;; close 1 finger
      ((equal *continuous-class* "tap_finger") ;; high_short
       (if (eq *state* :release)
         (progn
           (setq av-list (list (v+ (send *robot* :angle-vector) #f(0 0 0 -140))))
           (setq *state* :grasp-1-finger))
         (setq loop-continue nil)))
      ;; open 2 fingers
      ((equal *continuous-class* "rub_base") ;; low_long
       (if (eq *state* :grasp-2-finger)
         (progn
           (setq av-list (list (v+ (send *robot* :angle-vector) #f(0 0 0 130))
                               (v+ (send *robot* :angle-vector) #f(-150 0 0 130))))
           (setq *state* :release))
         (setq loop-continue nil)))
      ;; close 2 fingers
      ((equal *continuous-class* "tap_base") ;; low_short
       (if (eq *state* :release)
         (progn
           (setq av-list (list (v+ (send *robot* :angle-vector) #f(150 0 0 0))
                               (v+ (send *robot* :angle-vector) #f(150 0 0 -130))))
           (setq *state* :grasp-2-finger))
         (setq loop-continue nil)))
       ;; If *continuous-class* is none or nil, do nothing
       (t
        (setq loop-continue nil)))
     (when loop-continue
       (send *ri* :angle-vector-sequence av-list 2000)
       (send *robot* :angle-vector (car (last av-list)))
       (format t "Moving... (state is ~A)~%" *continuous-class*)
       (unix:sleep (+ 8 (* 4 (length av-list))))
       (send *ri* :wait-interpolation)
       (format t "Wait for input...~%")))
   (ros::sleep)
   ))
