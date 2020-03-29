(require "package://decopin_hand/euslisp/decopin_hand.l")

(defmethod decopin_hand-robot
  (:inverse-kinematics
   (target-coords &rest args &key link-list move-arm move-target &allow-other-keys)
   (unless move-target (setq move-target (send self :rarm :end-coords)))
   (unless link-list (setq link-list (send self :link-list (send move-target :parent))))
   (send-super* :inverse-kinematics target-coords
                :move-target move-target
                :link-list link-list
                args))
  )

(defun ik-example ()
  (send *robot* :reset-pose)
  (send *robot* :inverse-kinematics (make-coords :pos #f(-30 100 150)) :rotation-axis nil :revert-if-fail nil)
  (send *robot* :rarm :end-coords))
