#!/usr/bin/env roseus

(require "package://decopin_hand/euslisp/decopin_hand_utils.l")
(require "package://decopin_hand/euslisp/demo/hammering.l")
(require "package://fetcheus/fetch-interface.l")
(require "package://jsk_fetch_startup/euslisp/navigation-utils.l")
(ros::load-ros-manifest "decopin_hand")

;; Usage
;; In your PC
;; 1. roslaunch decopin_hand bringup_remote.launch
;; 2. rviz -d $(rospack find decopin_hand)/config/fetch.rviz
;; (Optional) roslaunch decopin_hand decopin_hand_kontrol.launch
;; 3. roslaunch decopin_hand classify_action.launch use_rosbag:=false gui:=true
;; (Optional) roslaunch decopin_hand record_audio_rosbag.launch use_fetch:=true
;; In fetch
;; 4. ssh yamaguchi@fetch15
;; 5. source /home/fetch/ros/indigo/devel/setup.bash
;; 6. roslaunch color_filter.launch


;; fetch : *fetch* and *ri*
;; decopin_hand : *robot* and *ri-hand*
(fetch-init)
(unless (boundp '*robot*)
  (setq *robot* (instance decopin_hand-robot :init)))
(unless (boundp '*ri-hand*)
  (setq *ri-hand* (instance decopin-hand-interface :init)))

;; Util function for *ri-hand*
;; Close gripper: (+ 0 + -)
;; Open gripper : (- 0 - +)
(defun move-gripper (diff-av &optional (tm 3000) &key (wait t))
  (send *ri-hand* :angle-vector
        (send *robot* :angle-vector
              (v+ (send *robot* :angle-vector) diff-av))
        tm)
  (if wait
    (send *ri-hand* :wait-interpolation)))

(defun main()
  ;; 1. Receive trash from human
  ;; 2. Classify trash
  ;; 3. Go in front of trash box
  ;; 4. Color recognition and throw the trash away to the correct trash box

  (receive-trash)
  (classify-trash)
  (go-to-trash-box)
  (throw-trash-away)
  )

(defun receive-trash ()
  (send *fetch* :reset-pose)
  (send *fetch* :inverse-kinematics
        (make-coords :pos #f(700 0 700))
        :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
  (move-gripper #f(-40 0 0 40)) ;; open
  (move-gripper #f(40 0 0 -40)) ;; close
  )

(defun classify-trash ()
  ;; Please see hammering.l
  )

(defun go-to-trash-box ()
  (send *ri* :move-to
        (make-coords :pos #f(3526 4000 0) :rpy (float-vector 0 0 -pi))
        :frame-id "/map")
  )

(defun throw-trash-away ()
  (ros::wait-for-service "/cloud_passthrough/request")
  (ros::service-call "/cloud_passthrough/request"
                     (instance std_srvs::EmptyRequest :init))
  (setq bb-list (one-shot-subscribe
                 "/HSI_color_filter/boxes" jsk_recognition_msgs::BoundingBoxArray))
  (ros::wait-for-service "/cloud_passthrough/stop")
  (ros::service-call "/cloud_passthrough/stop"
                     (instance std_srvs::EmptyRequest :init))
  ;; Calc position of trash from bb-list
  )
