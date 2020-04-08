#!/usr/bin/env roseus

(load "package://decopin_hand/euslisp/decopin_hand_interface.l")
(ros::load-ros-manifest "sensor_msgs")

(decopin-hand-init)

(setq *can-move* nil)
(setq *prev-av* #f(0 0 0 0))
(send *robot* :angle-vector
      (send *ri* :state :potentio-vector :wait-until-update t)))
(setq *last-moving-time* nil)

;; Usage
;; This robot hand has two mode: initial mode and working mode.
;; In initial mode, we decide initial position of robot hand. (*can-move* is nil)
;; In working mode, we move robot hand. (*can-move* is t)

;; We use several buttons to control this robot hand.
;; SET button: initial mode -> working mode
;; STOP button: working mode -> initial mode
;; START button: send control commands to real robot hand.
;; Sliders: selecting position of robot hand.

;; 6. (advanced) use roseus-smach. make-class and make :start, :set, :stop functions
;; 3. edit the above document and place docstring

(defun joy-cb (msg)
  (let* ((axes (send msg :axes))
         (ffinger (elt axes 8))
         (finger-rot (elt axes 9))
         (lfinger (elt axes 10))
         (rfinger (elt axes 11))
         (buttons (send msg :buttons))
         (set (elt buttons 3))
         (stop (elt buttons 8))
         (start (elt buttons 9))
         (av (send *robot* :angle-vector)))
    (if (eq *can-move* nil)
      (progn
        (when (eq set 1)
          (when (or (eq ffinger 0.0)
                    (eq finger-rot 0.0)
                    (eq lfinger 0.0)
                    (eq rfinger 0.0))
            (ros::ros-info "Please move 4 left sliders and then push SET button~%")
            (return-from joy-cb))
          (ros::ros-info "SET button is pushed. The initial pose of robot hand is set.~%")
          (setq *prev-av* (float-vector ffinger finger-rot lfinger rfinger))
          (return-from joy-cb))
        (when (eq start 1)
          (ros::ros-info "START button is pushed. The robot is going to move.~%")
          (setq *can-move* t)
          (setq *last-moving-time* nil)
          (return-from joy-cb))
        (return-from joy-cb))
      (progn
        (when (eq stop 1)
          (ros::ros-info "STOP button is pushed. You can move sliders freely.~%")
          (setq *can-move* nil)
          (return-from joy-cb))
        (when (send *ri* :interpolatingp)
          (ros::ros-error "Robot is moving. Command ignored.~%")
          (return-from joy-cb))
        (let* ((tm-now (ros::time-now))
               (current-av (float-vector ffinger finger-rot lfinger rfinger))
               (diff-av (scale 300 (v- current-av *prev-av*))))
          (when (eq *last-moving-time* nil)
            (setq *last-moving-time* tm-now)
            (return-from joy-cb))
          (setf (elt diff-av 3) (* -1 (elt diff-av 3))) ;; change direction of finger move
          (send *robot* :angle-vector (v+ (send *robot* :angle-vector) diff-av))
          (setq *prev-av* current-av)
          (setq *last-moving-time* tm-now)
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
   (when *last-moving-time*
     (let ((tm-now (ros::time-now)))
       (if (< (ros-time-elapsed tm-now *last-moving-time*) 0.3)
         (ros::ros-info "Receiving commands ...")
         (progn
           (ros::ros-info "Send angle-vector:~%~A~%" (send *robot* :angle-vector))
           (send *ri* :angle-vector (send *robot* :angle-vector) 500)
           (send *ri* :wait-interpolation)
           (ros::ros-info "You can send next angle vector~%" (send *robot* :angle-vector))
           (setq *last-moving-time* nil)))))
   (ros::sleep))
  (exit))


(main)
