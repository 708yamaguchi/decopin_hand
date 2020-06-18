;; Example use:
;; 1. simply collecting train data
;; (progn (move-hand-z -40) (unix:sleep 3) (hand-over-table :tm 3000))
;; 2. collecting interactive learning sample
;; (progn (move-hand-z -30) (unix:sleep 1) (hand-over-table :tm 2000) (move-hand-z -40) (unix:sleep 1) (hand-over-table :tm 2000))

(load "package://fetcheus/fetch-interface.l")

(unless (boundp '*fetch*)
  (fetch))
(unless (boundp '*ri*)
  (fetch-init))

(defun hand-over-table (&key (tm 10000))
  (send *fetch* :reset-pose)
  (send *fetch* :torso :waist-z :joint-angle 100)
  (send *fetch* :inverse-kinematics
        (make-coords :pos #f(700 0 850) :rpy (float-vector 0 0 pi/2)) :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) tm)
  (send *ri* :wait-interpolation))


(defun move-hand-z (move &key (tm 2000))
  (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
  (send *fetch* :rarm :move-end-pos (float-vector 0 0 move) :world :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) tm)
  (send *ri* :wait-interpolation))
