#!/usr/bin/env roseus

(require "package://decopin_hand/euslisp/decopin_hand_utils.l")
(ros::load-ros-manifest "jsk_recognition_msgs")

(decopin-hand-init)
(send *robot* :angle-vector (send *ri* :state :potentio-vector))
;; States are :release, :grasp-1-finger, :grasp-2-finger
;; Make sure that the hand is opening at first
(setq *state* :release)
(setq *dead-time-started* nil)
(setq *last-gesture* "none")
(setq *hand-can-move* nil)

;; If you want to filter *continuous-class* for gesture recognition
;; Edit (recognize-gesture) function
(defun recognize-gesture (continuous-class)
  (let ((ret continuous-class))
    ;; Check if now is dead time
    (when *dead-time-started*
      ;; Return the same gesture for a constant time for human visibility
      (if (< (- (send (ros::time-now) :to-sec) *dead-time-started*) 0.5)
        (return-from recognize-gesture *last-gesture*))
      (if (< (- (send (ros::time-now) :to-sec) *dead-time-started*) 10)
        (return-from recognize-gesture "dead_time")
        (setq *dead-time-started* nil)))
    ;; Recognize gesture
    (cond
     ;; open 1 finger
     ((equal *continuous-class* "rub_finger") ;; high_long
      (if (not (eq *state* :grasp-1-finger))
        (setq ret "none")))
     ;; close 1 finger
     ((equal *continuous-class* "tap_finger") ;; high_short
      (if (not (eq *state* :release))
        (setq ret "none")))
     ;; open 2 fingers
     ((equal *continuous-class* "rub_base") ;; low_long
      (if (not (eq *state* :grasp-2-finger))
        (setq ret "none")))
     ;; close 2 fingers
     ((equal *continuous-class* "tap_base") ;; low_short
      (if (not (eq *state* :release))
        (setq ret "none")))
     )
    ret))


(defun main ()
  (print "Please start demo")
  (ros::rate 30)
  (do-until-key
   (ros::spin-once)
   (let ((av-list nil) (gesture (recognize-gesture *continuous-class*)))
     ;; Check gesture. When now is dead_time or none, do not move hand
     (if (or (equal gesture "dead_time")
             (equal gesture "none"))
       (progn
         (setq *hand-can-move* t)
         (ros::sleep))
       (progn
         ;; Set hand direction
         (if (null *dead-time-started*)
           (setq *dead-time-started* (send (ros::time-now) :to-sec)))
         (cond
          ;; open 1 finger
          ((equal gesture "rub_finger") ;; high_long
           (setq av-list (list (v+ (send *robot* :angle-vector) #f(0 0 0 140))))
           (setq *state* :release))
          ;; close 1 finger
          ((equal gesture "tap_finger") ;; high_short
           (setq av-list (list (v+ (send *robot* :angle-vector) #f(0 0 0 -140))))
           (setq *state* :grasp-1-finger))
          ;; open 2 fingers
          ((equal gesture "rub_base") ;; low_long
           (setq av-list (list (v+ (send *robot* :angle-vector) #f(0 0 0 130))
                               (v+ (send *robot* :angle-vector) #f(-150 0 0 130))))
           (setq *state* :release))
          ;; close 2 fingers
          ((equal gesture "tap_base") ;; low_short
           (setq av-list (list (v+ (send *robot* :angle-vector) #f(150 0 0 0))
                               (v+ (send *robot* :angle-vector) #f(150 0 0 -130))))
           (setq *state* :grasp-2-finger)))
         ;; Move hand once.
         ;; We use *hand-can-move* flag to avoid using :wait-interpolation,
         ;; because we need constant publishing /gesture
         (when *hand-can-move*
           (setq *hand-can-move* nil)
           (send *ri* :angle-vector-sequence av-list 2000)
           (send *robot* :angle-vector (car (last av-list)))
           (format t "Moving... (recognized gesture is ~A)~%" gesture)
           (setq *last-gesture* gesture)
           ;; (send *ri* :wait-interpolation)
           (format t "Wait for input...~%"))))
   (ros::sleep))
   ))
