#!/usr/bin/env roseus

(load "package://decopin_hand/euslisp/decopin_hand_interface.l")
(ros::load-ros-manifest "sensor_msgs")

(decopin-hand-init)

(setq *can-move* nil)
(setq *initial-finger-rot* 0)
(setq *initial-finger1* 0)
(setq *initial-finger2* 0)
(setq *initial-finger3* 0)
(send *robot* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
;; (setq *av-initial* (send *robot* :angle-vector))
(setq *last-moving-command* nil)

;; Usage
;; This robot hand has two mode: initial mode and working mode.
;; In initial mode, we decide initial position of robot hand. (*can-move* is nil)
;; In working mode, we move robot hand. (*can-move* is t)

;; We use several buttons to control this robot hand.
;; SET button: initial mode -> working mode
;; STOP button: working mode -> initial mode
;; START button: send control commands to real robot hand.
;; Sliders: selecting position of robot hand.

;; TODO
;; 1. fix ros::ros-xxx message
;; 2. replace *initial-finger-xx* with *prev-commands*
;; 3. edit the above document and place docstring
;; 4. fix order of angle-vector
;; 5. fix direction of slider and finger movement
;; 6. (advanced) use roseus-smach. make-class and make :start, :set, :stop functions

(defun joy-cb (msg)
  (let* ((axes (send msg :axes))
         (finger-rot (elt axes 8))
         (finger1 (elt axes 9))
         (finger2 (elt axes 10))
         (finger3 (elt axes 11))
         (buttons (send msg :buttons))
         (set (elt buttons 3))
         (stop (elt buttons 8))
         (start (elt buttons 9))
         (av (send *robot* :angle-vector)))
    (when (or (eq finger-rot 0.0)
              (eq finger1 0.0)
              (eq finger2 0.0)
              (eq finger3 0.0))
      (ros::ros-info "Please move 4 left sliders and then push SET button~%")
      (return-from joy-cb))
    (if (eq *can-move* nil)
      (progn
        (when (eq set 1)
          (ros::ros-info "SET button is pushed. The initial pose of robot hand is set.~%")
          (setq *initial-finger-rot* finger-rot)
          (setq *initial-finger1* finger1)
          (setq *initial-finger2* finger2)
          (setq *initial-finger3* finger3)
          (return-from joy-cb))
        (when (eq start 1)
          (ros::ros-info "START button is pushed. The robot is going to move.~%")
          (setq *can-move* t)
          (setq *last-moving-command* nil)
          (return-from joy-cb))
        (ros::ros-error "Push SET button before moving hand.~%")
        (return-from joy-cb))
      (progn
        (when (eq stop 1)
          (ros::ros-info "STOP button is pushed. You can move sliders freely.~%")
          (setq *can-move* nil)
          (return-from joy-cb))
        (when (send *ri* :interpolatingp)
          (ros::ros-error "Robot is moving. Command ignored.~%")
          (return-from joy-cb))
        (let ((tm-now (ros::time-now))
              (diff-av
               (scale 300 (float-vector (- finger-rot *initial-finger-rot*)
                                        (- finger1 *initial-finger1*)
                                        (- finger2 *initial-finger2*)
                                        (- finger3 *initial-finger3*)))))
          (when (eq *last-moving-command* nil)
            (setq *last-moving-command* tm-now)
            (return-from joy-cb))
          (send *robot* :angle-vector (v+ (send *robot* :angle-vector) diff-av))

          (setq *initial-finger-rot* finger-rot)
          (setq *initial-finger1* finger1)
          (setq *initial-finger2* finger2)
          (setq *initial-finger3* finger3)

          (setq *last-moving-command* tm-now)
          )))))

(defun ros-time-elapsed (tm-after tm-before)
  (+ (- (send tm-after :sec) (send tm-before :sec))
     (* 10e-9 (- (send tm-after :nsec) (send tm-before :nsec)))))

(defun main ()
  (ros::subscribe "/nanokontrol/joy" sensor_msgs::Joy #'joy-cb)
  (ros::rate 100)
  (ros::ros-info "~%Please use nanoKONTROL2~%")
  (do-until-key
   (ros::spin-once)

   (when *last-moving-command*
     (let ((tm-now (ros::time-now)))
       (if (< (ros-time-elapsed tm-now *last-moving-command*) 0.3)
         (ros::ros-info "Receiving commands ...")
         (progn
           (ros::ros-info "Send angle-vector:~%~A" (send *robot* :angle-vector))
           (send *ri* :angle-vector (send *robot* :angle-vector) 500)
           (send *ri* :wait-interpolation)
           (setq *last-moving-command* nil)))))

   (ros::sleep))
  (exit))


(main)
