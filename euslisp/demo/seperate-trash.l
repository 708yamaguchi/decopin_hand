#!/usr/bin/env roseus

(require "package://decopin_hand/euslisp/decopin_hand_utils.l")
(require "package://decopin_hand/euslisp/demo/hammering.l")
(require "package://fetcheus/fetch-interface.l")
(require "package://jsk_fetch_startup/euslisp/navigation-utils.l")
(ros::load-ros-manifest "decopin_hand")
(ros::load-ros-manifest "speech_recognition_msgs")

;; Usage
;; In your PC
;; 0. rossetmaster fetch15
;; 1. roslaunch decopin_hand bringup_remote.launch # maybe 2 times
;; (Optional) roslaunch decopin_hand save_noise.launch
;; (Optional) roslaunch decopin_hand decopin_hand_kontrol.launch
;; 2. roslaunch decopin_hand classify_action.launch use_rosbag:=false gui:=true
;; 3. rviz -d $(rospack find decopin_hand)/config/fetch.rviz
;; In fetch
;; 4. ssh yamaguchi@fetch15
;; 5. source /home/fetch/ros/indigo/devel/setup.bash
;; 6. roslaunch color_filter.launch
;; (Optional) roslaunch decopin_hand record_audio_rosbag.launch use_fetch:=true

;; fetch : *fetch* and *ri*
;; decopin_hand : *robot* and *ri-hand*
(fetch-init)
(unless (boundp '*robot*)
  (setq *robot* (instance decopin_hand-robot :init)))
(unless (boundp '*ri-hand*)
  (setq *ri-hand* (instance decopin-hand-interface :init)))

;; speech recognition
(defun speech-cb (msg)
  (setq a msg)
  ;; (print (car (send msg :transcript)))
  (when (substringp "ゴミ" (car (send msg :transcript)))
    (print "I hear ゴミを捨てて")
    (setq *start-demo* t))
  )
(ros::subscribe "/speech_to_text_google"
                speech_recognition_msgs::SpeechRecognitionCandidates #'speech-cb)
(setq *start-demo* nil)

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

  (let (trash-class)
    (receive-trash)
    (setq trash-class (classify-trash))
    (go-to-trash-box)
    (throw-trash-away :left) ;; TODO: use the result of classify-trash
    )
  )

(defun receive-trash ()
  (format t "receive trash started~%")
  (reset-pose 5000)
  (send *ri* :move-to
        (make-coords :pos #f(4359 6408 0) :rpy (float-vector -pi 0 0))
        :frame-id "/map")
  ;; Start demo by speech recognition
  (ros::rate 3)
  (format t "Start demo when I hear ゴミを捨てて~%")
  (setq *start-demo* nil)
  (while t
    (ros::spin-once)
    (when *start-demo*
      (return))
    (ros::sleep))
  (send *ri* :speak-jp "ゴミを捨てに行きます")
  ;; Receive trash
  (send *fetch* :reset-pose)
  (send *fetch* :torso :waist-z :joint-angle 100)
  (send *fetch* :inverse-kinematics
        (make-coords :pos #f(700 0 800))
        :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
  (send *ri* :wait-interpolation)
  ;; TODO
  ;; (move-gripper #f(-40 0 0 40)) ;; open
  ;; (move-gripper #f(40 0 0 -40)) ;; close
  ;; TODO
  (format t "Receive trash ended~%")
  )

(defun classify-trash ()
  (format t "Classify trash started~%")
  ;; Please see hammering.l
  (set-pose 7000)
  (unix:sleep 1)
  (send *fetch* :head :neck-y :joint-angle
        (- (send *fetch* :head :neck-y :joint-angle ) 25))
  (send *ri* :angle-vector (send *fetch* :angle-vector) 1000)
  (while (null (send *ri* :interpolatingp))
    (unix:usleep (* 100 1000))) ;; wait for interpolatingp become t
  (let ((trash-class "none") (trash-class-ja))
    (while (send *ri* :interpolatingp)
      (ros::spin-once)
      (if (equal *continuous-class* "none")
        (unix:usleep (* 100 1000))
        (progn
          (setq trash-class *continuous-class*)
          (return))))
    (format t "trash-class: ~A~%" trash-class)
    (reset-pose 7000)
    (format t "Classify trash ended~%")
    (when (equal trash-class-ja "can")
      (setq trash-class-ja "缶"))
    (when (equal trash-class-ja "bottle")
      (setq trash-class-ja "ペットボトル"))
    (send *ri* :speak-jp (format nil "これは~Aです。" trash-class-ja) :wait t)
    trash-class)
  )

(defun go-to-trash-box ()
  (send *ri* :move-to
        (make-coords :pos #f(2300 5320 0) :rpy (float-vector -pi 0 0))
        :frame-id "/map")
  )

(defun find-trash-box ()
  ;; Calc position of trash from bb-list
  ;; Navigition is not precise, so look around
  (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
  (send *fetch* :head :neck-p :joint-angle 35)
  (send *ri* :angle-vector (send *fetch* :angle-vector) 1000)
  (format t "Look down~%")
  (send *ri* :wait-interpolation)
  (ros::wait-for-service "/cloud_passthrough/request")
  (ros::service-call "/cloud_passthrough/request"
                     (instance std_srvs::EmptyRequest :init))
  (let (bb-list left-bb-pos right-bb-pos tmp)
    (setq bb-list (one-shot-subscribe
                   "/HSI_color_filter/boxes" jsk_recognition_msgs::BoundingBoxArray))
    (unless (eq (length (send bb-list :boxes)) 2)
      (send *fetch* :head :neck-y :joint-angle 30)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 1000)
      (format t "Look left~%")
      (send *ri* :wait-interpolation)
      (setq bb-list (one-shot-subscribe
                     "/HSI_color_filter/boxes" jsk_recognition_msgs::BoundingBoxArray)))
    (unless (eq (length (send bb-list :boxes)) 2)
      (send *fetch* :head :neck-y :joint-angle -30)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 1000)
      (format t "Look right~%")
      (send *ri* :wait-interpolation)
      (setq bb-list (one-shot-subscribe
                     "/HSI_color_filter/boxes" jsk_recognition_msgs::BoundingBoxArray)))
    (if (eq (length (send bb-list :boxes)) 2)
      (format t "Find 2 trash boxes!~%")
      (format t "Cannot find 2 trash boxes ...~%"))
    (setq left-bb-pos
          (send (ros::tf-pose->coords (send (car (send bb-list :boxes)) :pose)) :pos))
    (setq right-bb-pos
          (send (ros::tf-pose->coords (send (cadr (send bb-list :boxes)) :pose)) :pos))
    (if (< (elt left-bb-pos 1) (elt right-bb-pos 1))
      (progn
        (setq tmp left-bb-pos)
        (setq left-bb-pos right-bb-pos)
        (setq right-bb-pos left-bb-pos)))
    (format t "left-bb-pos: ~A right-bb-pos: ~A~%" left-bb-pos right-bb-pos)
    (ros::wait-for-service "/cloud_passthrough/stop")
    (ros::service-call "/cloud_passthrough/stop"
                       (instance std_srvs::EmptyRequest :init))
    (list left-bb-pos right-bb-pos))
  )

;; TODO currently, left-bb is the target
(defun throw-trash-away (which-box)
  (format t "Throw trash away started~%")
  (let* ((neck-p (send *fetch* :head :neck-p :joint-angle))
         (neck-y (send *fetch* :head :neck-y :joint-angle))
         (bbs (find-trash-box))
         (left-bb-pos (car bbs))
         (right-bb-pos (cadr bbs))
         target-bb-pos target-coords1 target-coords2)
    ;; left-bb-pos's example: (setq left-bb-pos #f(833.763 399.305 285.967))
    (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
    (send *fetch* :torso :waist-z :joint-angle 100)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    (send *fetch* :reset-pose)
    (send *fetch* :torso :waist-z :joint-angle 100)
    (setq target-coords1
          (make-coords :pos #f(600 0 1000) :rpy (float-vector pi/2 0 0)))
    (cond
     ((eq which-box :left)
      (setq target-bb-pos left-bb-pos))
     ((eq which-box :right)
      (setq target-bb-pos right-bb-pos))
     ((eq which-box :center)
      (setq target-bb-pos (midpoint 0.5 #f(0 0 0) #f(100 100 100)left-bb-pos)))
     (t
      (format t "Argument of (throw-trash-away) is invalid !!")
      (return-from throw-trash-away)))

    (setq target-coords2
          (make-coords :pos (float-vector (elt target-bb-pos 0)
                                          (elt target-bb-pos 1)
                                          800)))
    (send *fetch* :inverse-kinematics target-coords1 :use-torso nil)
    (send *fetch* :head :neck-p :joint-angle neck-p)
    (send *fetch* :head :neck-y :joint-angle neck-y)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
    (send *ri* :wait-interpolation)
    (send *fetch* :inverse-kinematics target-coords2 :use-torso nil)
    (send *fetch* :head :neck-p :joint-angle neck-p)
    (send *fetch* :head :neck-y :joint-angle neck-y)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
    (send *ri* :wait-interpolation)

    ;; TODO
    ;; (move-gripper #f(-40 0 0 40)) ;; open
    ;; TODO

    (send *fetch* :inverse-kinematics target-coords1 :use-torso nil)
    (send *fetch* :head :neck-p :joint-angle neck-p)
    (send *fetch* :head :neck-y :joint-angle neck-y)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
    (send *ri* :wait-interpolation)
    (reset-pose 5000)
    (format t "Throw trash away ended~%")
  ))
