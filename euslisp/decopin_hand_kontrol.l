#!/usr/bin/env roseus

(require "package://roseus_smach/src/state-machine-ros.l")
(require "package://decopin_hand/euslisp/decopin_hand_interface.l")
(ros::load-ros-manifest "sensor_msgs")


(defclass decopin-hand-kontrol
  :super propertied-object
  :slots (can-move- reference-av- current-av- last-moving-time-
          pushed-button-)
  :documentation
  "
  You can control robot hand fingers via nanoKONTROL2 MIDI.
  There are 4 modes (START, SETUP, MOVING, IDLE), which are managed by state machine.
  START mode  : You need to move sliders and put SET button
  SETUP mode  : You can set reference angle-vector by SET button
  MOVING mode : You can move the fingers by moving sliders. The amount of movement is based on the reference angle-vector.
  IDLE mode   : Robot fingers do not move, so you can move the sliders freely.
  You can see the state transition by rosrun smach_viewer smach_viewer.py
  "
  )

(defmethod decopin-hand-kontrol
  (:init ()
    ;; Setup robot
    (decopin-hand-init)
    (send *robot* :angle-vector
          (send *ri* :state :potentio-vector :wait-until-update t))
    (setq reference-av- #f(0 0 0 0))
    (setq last-moving-time- nil)
    (setq can-move- nil)
    (setq pushed-button- :stop) ;; :stop, :start or :setup
    ;; Register subscriber
    (ros::subscribe "/nanokontrol/joy" sensor_msgs::Joy #'send self :joy-cb)
    (ros::ros-info "~%Please use nanoKONTROL2~%")
    self)
  (:idle-func (&rest args)
   (cond
    ((eq pushed-button- :setup)
     (setq reference-av- current-av-)
     (return-from :idle-func :setup-pushed))
    ((eq pushed-button- :start)
     (return-from :idle-func :start-pushed))
    (t
     (return-from :idle-func :stay))))
  (:setup-func (&rest args)
    (cond
     ((eq pushed-button- :start)
      (return-from :setup-func :start-pushed))
     ((eq pushed-button- :setup)
      (setq reference-av- current-av-)
      (return-from :setup-func :stay))
     (t
      (return-from :setup-func :stay))))
  (:moving-func (&rest args)
    (cond
     ((eq pushed-button- :stop)
      (return-from :moving-func :stop-pushed))
     (t
      (when last-moving-time-
        (let ((tm-now (ros::time-now))
              (diff-av (scale 300 (v- current-av- reference-av-))))
          (if (< (ros-time-elapsed tm-now last-moving-time-) 0.3)
            (ros::ros-info "Receiving commands ...")
            (progn
              (setq reference-av- current-av-)
              (setf (elt diff-av 3) (* -1 (elt diff-av 3))) ;; change direction of finger move
              (ros::ros-info "diff-av: ~A~%" diff-av)
              (send *robot* :angle-vector (v+ (send *robot* :angle-vector) diff-av))
              (send *ri* :angle-vector (send *robot* :angle-vector) 500)
              (send *ri* :wait-interpolation)
              (ros::ros-info "You can send next angle vector~%" (send *robot* :angle-vector))
              (setq last-moving-time- nil)))))
      (return-from :moving-func :stay))))
  (:joy-cb (msg) ()
    (let* ((axes (send msg :axes))
           (buttons (send msg :buttons))
           (set (elt buttons 3))
           (stop (elt buttons 8))
           (start (elt buttons 9)))
      (print "joy-cb")
      (setq current-av- (float-vector (elt axes 8) (elt axes 9) (elt axes 10) (elt axes 11)))
      (cond
       ((eq set 1)
        (when (or (eq 0.0 (elt current-av- 0))
                  (eq 0.0 (elt current-av- 1))
                  (eq 0.0 (elt current-av- 2))
                  (eq 0.0 (elt current-av- 3)))
          (ros::ros-error "Please move 4 left sliders and then push SET button~%")
          (return-from :joy-cb))
        (ros::ros-info "SET button is pushed. The initial pose of robot hand is set.~%")
        (setq pushed-button- :setup))
       ((eq stop 1)
        (ros::ros-info "STOP button is pushed. You can move sliders freely.~%")
        (setq pushed-button- :stop))
       ((eq start 1)
        (ros::ros-info "START button is pushed. The robot is going to move.~%")
        (setq pushed-button- :start)))
      (let* ((tm-now (ros::time-now)))
        (when (eq last-moving-time- nil)
          (setq last-moving-time- tm-now)
          (return-from :joy-cb))
        (setq last-moving-time- tm-now)))))


(defun ros-time-elapsed (tm-after tm-before)
  (+ (- (send tm-after :sec) (send tm-before :sec))
     (* 10e-9 (- (send tm-after :nsec) (send tm-before :nsec)))))


(defun main ()
  ;; Set nanoKONTROL interface
  (setq dhk (instance decopin-hand-kontrol :init))
  ;; Set state machine
  ;; TODO: delete setup mode?
  (setq sm (instance state-machine :init))
  (send sm :add-node (instance state :init :START
                               '(lambda (&rest args) (send dhk :idle-func args))))
  (send sm :add-node (instance state :init :SETUP
                               '(lambda (&rest args) (send dhk :setup-func args))))
  (send sm :add-node (instance state :init :MOVING
                               '(lambda (&rest args) (send dhk :moving-func args))))
  (send sm :add-node (instance state :init :IDLE
                               '(lambda (&rest args) (send dhk :idle-func args))))
  (send sm :add-transition :START :SETUP :setup-pushed)
  (send sm :add-transition :START :START :stay)
  (send sm :add-transition :SETUP :MOVING :start-pushed)
  (send sm :add-transition :SETUP :SETUP :stay)
  (send sm :add-transition :MOVING :IDLE :stop-pushed)
  (send sm :add-transition :MOVING :MOVING :stay)
  (send sm :add-transition :IDLE :MOVING :start-pushed)
  (send sm :add-transition :IDLE :SETUP :setup-pushed)
  (send sm :add-transition :IDLE :IDLE :stay)
  (send sm :start-state :START)
  (send sm :goal-state :no-goal)
  ;; Start control
  (exec-state-machine sm '(nil) :hz 100))
