//
//  ViewController.swift
//  ClassifyEverything
//
//  Created by TuyenVX on 3/27/18.
//  Copyright © 2018 TuyenVX. All rights reserved.
//

import UIKit
import Vision
import AVFoundation

class ViewController: UIViewController {
    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var answerLabel: UILabel!
    @IBOutlet weak var cameraButton: UIButton!

    var captureSession: AVCaptureSession?
    var rearCamera: AVCaptureDevice?
    var rearCameraInput: AVCaptureDeviceInput?
    var photoOutput: AVCapturePhotoOutput?
    var previewLayer: AVCaptureVideoPreviewLayer?
    var photoCaptureCompletionBlock: ((UIImage?, Error?) -> Void)?
    
    lazy var classificationRequest: VNCoreMLRequest = {
        do {

            let model = try VNCoreMLModel(for: Resnet50().model)
            
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    

    override func viewDidLoad() {
        super.viewDidLoad()
        config()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
    }
    
    // MARK: - Camera preview
    
    func config() {
        prepare {(error) in
            if let error = error {
                print(error)
            }
            try? self.displayPreview(on: self.cameraView)
        }
    }
    
    func displayPreview(on view: UIView) throws {
        guard let `captureSession` = captureSession, captureSession.isRunning else {
            throw CameraControllerError.captureSessionIsMissing
        }
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.videoGravity = .resizeAspectFill
        previewLayer?.connection?.videoOrientation = .portrait
        view.layer.insertSublayer(previewLayer!, at: 0)
        previewLayer?.frame = view.frame
    }

    func prepare(completionHandler: @escaping (Error?) -> Void) {
        func createCaptureSession() {
            self.captureSession = AVCaptureSession()
        }
        func configureCaptureDevices() throws {
            let session = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .unspecified)
            let cameras = (session.devices.flatMap { $0 })
            for camera in cameras {
                if camera.position == .back {
                    rearCamera = camera
                    try camera.lockForConfiguration()
                    camera.focusMode = .continuousAutoFocus
                    camera.unlockForConfiguration()
                }
            }
        }
        func configureDeviceInputs() throws {
            guard let `captureSession` = captureSession else {
                throw CameraControllerError.captureSessionIsMissing
            }
            guard let `rearCamera` = rearCamera else {
                throw CameraControllerError.noCamerasAvailable
            }
            rearCameraInput = try AVCaptureDeviceInput(device: rearCamera)
            if captureSession.canAddInput(rearCameraInput!) {
                captureSession.addInput(rearCameraInput!)
            }
        }
        func configurePhotoOutput() throws {
            guard let `captureSession` = captureSession else {
                throw CameraControllerError.captureSessionIsMissing
            }
            photoOutput = AVCapturePhotoOutput()
            photoOutput?.setPreparedPhotoSettingsArray([AVCapturePhotoSettings(format: [AVVideoCodecKey : AVVideoCodecType.jpeg])], completionHandler: nil)
            if captureSession.canAddOutput(photoOutput!) {
                captureSession.addOutput(photoOutput!)
            }
            captureSession.startRunning()
        }
        
        DispatchQueue(label: "prepare").async {
            do {
                createCaptureSession()
                try configureCaptureDevices()
                try configureDeviceInputs()
                try configurePhotoOutput()
            }
                
            catch {
                DispatchQueue.main.async {
                    completionHandler(error)
                }
                
                return
            }
            
            DispatchQueue.main.async {
                completionHandler(nil)
            }
        }
    }
    // MARK: - Capture Image
    func captureImage(completion: @escaping (UIImage?, Error?) -> Void) {
        guard let `captureSession` = captureSession, captureSession.isRunning else {
            completion(nil, CameraControllerError.captureSessionIsMissing)
            return
        }
        let setting = AVCapturePhotoSettings()
        photoOutput?.capturePhoto(with: setting, delegate: self)
        photoCaptureCompletionBlock = completion
    }
    @IBAction func didSelectedCameraButton(_ sender: UIButton) {
        captureImage { (image, error) in
            guard let image = image else {
                return
            }
            self.updateClassifications(for: image)
        }
    }
    
    // MARK: - Machine Learning
    
    /// - Tag: PerformRequests
    func updateClassifications(for image: UIImage) {
        answerLabel.text = "Classifying..."
        
        let orientation = CGImagePropertyOrientation(image.imageOrientation)
        guard let ciImage = CIImage(image: image) else { fatalError("Unable to create \(CIImage.self) from \(image).") }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            do {
                try handler.perform([self.classificationRequest])
            } catch {
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
    }
    
    /// Updates the UI with the results of the classification.
    /// - Tag: ProcessClassifications
    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            guard let results = request.results else {
                self.answerLabel.text = "Unable to classify image.\n\(error!.localizedDescription)"
                return
            }
            // The `results` will always be `VNClassificationObservation`s, as specified by the Core ML model in this project.
            let classifications = results as! [VNClassificationObservation]
            
            if classifications.isEmpty {
                self.answerLabel.text = "Nothing recognized."
            } else {
                // Display top classifications ranked by confidence in the UI.
                let topClassifications = classifications.prefix(2)
                let descriptions = topClassifications.map { classification in
                    // Formats the classification for display; e.g. "(0.37) cliff, drop, drop-off".
                    return String(format: "  (%.2f) %@", classification.confidence, classification.identifier)
                }
                self.answerLabel.text = "Classification:\n" + descriptions.joined(separator: "\n")
            }
        }
    }
}

extension ViewController: AVCapturePhotoCaptureDelegate {
    public func photoOutput(_ captureOutput: AVCapturePhotoOutput, didFinishProcessingPhoto photoSampleBuffer: CMSampleBuffer?, previewPhoto previewPhotoSampleBuffer: CMSampleBuffer?,
                        resolvedSettings: AVCaptureResolvedPhotoSettings, bracketSettings: AVCaptureBracketedStillImageSettings?, error: Swift.Error?) {
        if let error = error { self.photoCaptureCompletionBlock?(nil, error) }
            
        else if let buffer = photoSampleBuffer, let data = AVCapturePhotoOutput.jpegPhotoDataRepresentation(forJPEGSampleBuffer: buffer, previewPhotoSampleBuffer: nil),
            let image = UIImage(data: data) {
            
            self.photoCaptureCompletionBlock?(image, nil)
        }
            
        else {
            self.photoCaptureCompletionBlock?(nil, CameraControllerError.unknown)
        }
    }
}
enum CameraControllerError: Swift.Error {
    case captureSessionAlreadyRunning
    case captureSessionIsMissing
    case inputsAreInvalid
    case invalidOperation
    case noCamerasAvailable
    case unknown
}
