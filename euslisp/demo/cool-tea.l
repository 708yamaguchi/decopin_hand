#!/usr/bin/env roseus

(require "package://decopin_hand/euslisp/decopin_hand_utils.l")
;; (require "package://decopin_hand/euslisp/demo/hammering.l")
(require "package://fetcheus/fetch-interface.l")
(ros::load-ros-manifest "decopin_hand")

;; Usage
;; In your PC
;; 0. rossetmaster fetch15
;; 1. roslaunch decopin_hand bringup_remote.launch # maybe 2 times
;; (Optional) roslaunch decopin_hand save_noise.launch
;; (Optional) roslaunch decopin_hand decopin_hand_kontrol.launch
;; 2. roslaunch decopin_hand classify_action.launch use_rosbag:=false gui:=true
;; (Optional) ssh raspi708
;; (Optional) roslaunch decopin_hand record_audio_rosbag.launch use_fetch:=false filename:=/home/naoya/hoge.bag

;; fetch : *fetch* and *ri*
;; decopin_hand : *robot* and *ri-hand*
(fetch-init)
(unless (boundp '*robot*)
  (setq *robot* (instance decopin_hand-robot :init)))
(unless (boundp '*ri-hand*)
  (setq *ri-hand* (instance decopin-hand-interface :init)))

(defun reset-pose (&optional (tm 5000))
  (send *fetch* :reset-pose)
  (send *fetch* :rarm :move-end-pos #f(200 -200 0) :world :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) tm)
  (send *ri* :wait-interpolation))

;; Util function for *ri-hand*
;; Close gripper: (+ 0 + -)
;; Open gripper : (- 0 - +)
;; *ri-hand* version
(defun move-gripper (diff-av &optional (tm 3000) &key (wait t))
  (send *ri-hand* :angle-vector
        (send *robot* :angle-vector
              (v+ (send *robot* :angle-vector) diff-av))
        tm)
  (if wait
    (send *ri-hand* :wait-interpolation)))

(defun torso-up (up)
  (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
  (send *fetch* :torso :waist-z :joint-angle
        (+ up (send *fetch* :torso :waist-z :joint-angle)))
  (send *ri* :angle-vector (send *fetch* :angle-vector) 3000))

(defun set-pose (&optional (tm 5000) (send-ri t))
  (send *fetch* :reset-pose)
  (send *fetch* :torso :waist-z :joint-angle 250)
  (send *fetch* :inverse-kinematics
        (make-coords :pos (float-vector 700 0 1000) :rpy (float-vector 0 0 0))
        :debug-view :no-message
        :min-loop 50
        :revert-if-fail nil
        :additional-nspace-list (mapcar #'(lambda (link refangle)
                                            (eval `(list link #'(lambda () (* 0.1 (- ,refangle (send ,link :joint :joint-angle)))))))
                                        (send-all (list (nth 2 (send *fetch* :rarm :joint-list))) :child-link)
                                        (list -20))
        :use-torso nil)
  (send *fetch* :torso :waist-z :joint-angle
        (+ -80 (send *fetch* :torso :waist-z :joint-angle)))
  (send *fetch* :rarm :wrist-r :joint-angle
        (+ 60 (send *fetch* :rarm :wrist-r :joint-angle)))
  (when send-ri
    (send *ri* :angle-vector (send *fetch* :angle-vector) tm)
    (send *ri* :wait-interpolation))
  )

(defun search-for-tea-cup ()
  (format t "search-for-tea-cup~%")
  (send *fetch* :rarm :move-end-pos #f(0 100 0) :world :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) 1000)
  (while (null (send *ri* :interpolatingp))
    (unix:usleep (* 100 1000))) ;; wait for interpolatingp become t
  (let ((find-cup nil) (start-time (send (ros::time-now) :to-sec)))
    ;; (while (send *ri* :interpolatingp)
    (while (< (- (send (ros::time-now) :to-sec) start-time) 1.5)
      (ros::spin-once)
      ;; (if (equal *continuous-class* "touch")
      (if (equal *temporary-class* "touch")
        (progn
          (setq find-cup t)
          (return))
        (unix:usleep (* 100 1000))))
    (send *ri* :cancel-angle-vector)
    (unless find-cup
      (format t "Failed to find tea cup~%")
      (return-from search-for-tea-cup)))
  (format t "Find tea cup !!~%")
  )

(defun stir ()
  (format t "stir~%")
  (let (av0 av1 av2 av3 av4)
    (setq av0 (set-pose 4000 nil)) ;; do not (send *ri* :angle-vector)
    (send *fetch* :angle-vector av0)
    (setq av1 (send *fetch* :rarm :move-end-pos #f(0 0 100) :world :use-torso nil))
    (setq av2 (send *fetch* :rarm :move-end-pos #f(100 100 0) :world :use-torso nil))
    (setq av3 (send *fetch* :rarm :move-end-pos #f(100 -100 0) :world :use-torso nil))
    (setq av4 (send *fetch* :rarm :move-end-pos #f(-100 -100 0) :world :use-torso nil))
    (dotimes (i 4) ;; about 30 seconds stir
      (send *ri* :angle-vector-sequence (list av1 av2 av3 av4) (list 2000 2000 2000 2000))
      (unix:sleep 9)
      )
    ))

(defun present-human ()
  (format t "present-human~%")
  (send *fetch* :rarm :move-end-pos #f(0 300 100) :world :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (let ((find-human nil))
    (do-until-key
     (ros::spin-once)
     (if (equal *continuous-class* "touch")
       (progn
         (setq find-human t)
         (return))
       (unix:usleep (* 100 1000))))
    (unless find-human
      (format t "Failed to find human~%")
      (return-from present-human)))
  (move-gripper #f(-70 0 0 70))
  )

;; before this function, please set-pose in front of the kitchen
;; gripper fingers angle-vector starts like #f(0 0 0 -90)
(defun main ()
  (set-pose)
  ;; Search for tea cup
  (search-for-tea-cup)
  ;; grasp and cool tea
  (move-gripper #f(70 0 0 -40))
  (stir)
  ;; present human
  (present-human)
  )
