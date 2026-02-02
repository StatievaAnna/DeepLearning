from face_detection import FaceDetection


def main():
    same_dir = 'data/tony_stark'
    diff_dir = 'data/other_people'
    anchor_image_path = 'data/anchor/tony_stark2.png'
    
    face_worker = FaceDetection()

    distances, labels = face_worker.load_pairs(anchor_image_path, same_dir, diff_dir)
    threshold = face_worker.find_best_threshold(distances, labels)
    face_worker.plot_roc(distances, labels, name='Robert Downey')
    

if __name__ == "__main__":
    main()