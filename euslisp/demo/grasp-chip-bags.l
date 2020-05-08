#!/usr/bin/env roseus

(require "package://decopin_hand/euslisp/decopin_hand_utils.l")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "speech_recognition_msgs")

(decopin-hand-init)
(send *robot* :angle-vector (send *ri* :state :potentio-vector))


;; for collecting train data
(defun close-gripper (&key (wait t))
  (move-gripper #f(15 0 0 -15) 1000 :wait wait)
  )
(defun open-gripper (&key (wait t))
  (move-gripper #f(-15 0 0 15) 1000 :wait wait)
  )

(defun speech-cb (msg)
  (setq a msg)
  (when (equal (car (send msg :transcript)) (format nil "掴んで"))
    (print "I hear 掴んで")
    (setq *speech-result* t))
  )

(defun cb (msg)
  (setq a msg)
  (when (equal (car (send msg :label_names)) (format nil "chip_bag~%"))
    (print "find chip bag !!")
    (setq *classify-result* t))
  )


(defun main ()
  (ros::subscribe "/Tablet/voice"
                  speech_recognition_msgs::SpeechRecognitionCandidates #'speech-cb)
  (ros::subscribe "/action_classifier/output"
                  jsk_recognition_msgs::ClassificationResult #'cb)

  ;; speech recognition before start demo
  ;; (ros::rate 10)
  ;; (setq *speech-result* nil)
  ;; (print "Start speech recognition")
  ;; (do-until-key
  ;;  (ros::spin-once)
  ;;  (when *speech-result*
  ;;    (return)))
  ;; (print "OK. Start grasping ...")
  ;; (setq *speech-result* nil)

  ;; grasping
  (ros::rate 100)
  (setq *classify-result* nil)
  (move-gripper #f(50 0 0 -50) 5000 :wait nil)
  (do-until-key
   (ros::spin-once)
   (when *classify-result*
     (send *ri* :stop-motion)
     (return)))
  (print "I grasped chip bag.")
  (setq *classify-result* nil)
  )
