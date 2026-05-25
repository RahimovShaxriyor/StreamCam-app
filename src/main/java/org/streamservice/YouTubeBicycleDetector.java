package org.streamservice;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegLogCallback;
import org.bytedeco.javacv.Java2DFrameConverter;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import static org.bytedeco.ffmpeg.global.avutil.AV_LOG_QUIET;

public class YouTubeBicycleDetector {

    private static final String YOUTUBE_URL = "https://www.youtube.com/watch?v=8JCk5M_xrBs";
    private static final String MODEL_PATH = "yolo11n.onnx";

    // Для real-time лучше 416, не 640
    private static final int INPUT_SIZE = 416;

    // COCO classes: 0 = person, 1 = bicycle, 2 = car
    private static final int BICYCLE_CLASS_ID = 1;

    private static final float CONF_THRESHOLD = 0.45f;
    private static final float NMS_THRESHOLD = 0.45f;

    // YOLO будет работать примерно 4 раза в секунду
    private static final int DETECTION_INTERVAL_MS = 250;

    // Видео будет обновляться примерно 30 FPS
    private static final int DISPLAY_INTERVAL_MS = 33;

    public static void main(String[] args) throws Exception {
        // Убираем лишние FFmpeg-логи
        FFmpegLogCallback.setLevel(AV_LOG_QUIET);

        System.out.println("Getting low-latency stream URL...");
        String streamUrl = getYouTubeStreamUrl(YOUTUBE_URL);

        System.out.println("Loading ONNX model...");

        OrtEnvironment env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setIntraOpNumThreads(4);
        options.setInterOpNumThreads(1);

        OrtSession session = env.createSession(MODEL_PATH, options);
        String inputName = session.getInputNames().iterator().next();

        System.out.println("Model loaded.");
        System.out.println("Starting real-time mode...");

        AtomicBoolean running = new AtomicBoolean(true);

        AtomicReference<BufferedImage> latestFrame = new AtomicReference<>();
        AtomicReference<List<Detection>> latestDetections =
                new AtomicReference<>(Collections.emptyList());

        AtomicLong frameSequence = new AtomicLong(0);

        ImagePanel panel = new ImagePanel();

        JFrame window = new JFrame("Real-Time Bicycle Detection");
        window.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        window.setSize(1000, 700);
        window.setLocationRelativeTo(null);
        window.add(panel);
        window.setVisible(true);

        window.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                running.set(false);
            }

            @Override
            public void windowClosed(WindowEvent e) {
                running.set(false);
            }
        });

        Thread captureThread = new Thread(() -> {
            FFmpegFrameGrabber grabber = null;
            Java2DFrameConverter converter = new Java2DFrameConverter();

            try {
                grabber = createLowLatencyGrabber(streamUrl);
                grabber.start();

                while (running.get()) {
                    org.bytedeco.javacv.Frame grabbed = grabber.grabImage();

                    if (grabbed == null) {
                        sleep(5);
                        continue;
                    }

                    BufferedImage image = converter.convert(grabbed);

                    if (image == null) {
                        continue;
                    }

                    /*
                     * Главное:
                     * мы НЕ складываем кадры в очередь.
                     * Мы всегда заменяем старый кадр новым.
                     */
                    latestFrame.set(image);
                    frameSequence.incrementAndGet();
                }

            } catch (Exception e) {
                System.err.println("Capture error: " + e.getMessage());
                running.set(false);
            } finally {
                try {
                    if (grabber != null) {
                        grabber.stop();
                        grabber.release();
                    }
                } catch (Exception ignored) {
                }
            }
        }, "capture-thread");

        Thread detectionThread = new Thread(() -> {
            long lastProcessedSequence = -1;

            while (running.get()) {
                long start = System.currentTimeMillis();

                try {
                    long currentSequence = frameSequence.get();

                    if (currentSequence != lastProcessedSequence) {
                        BufferedImage frame = latestFrame.get();

                        if (frame != null) {
                            List<Detection> detections =
                                    detectBicycles(frame, env, session, inputName);

                            latestDetections.set(detections);
                            lastProcessedSequence = currentSequence;
                        }
                    }

                } catch (Exception e) {
                    System.err.println("Detection error: " + e.getMessage());
                }

                long elapsed = System.currentTimeMillis() - start;
                long sleepTime = Math.max(5, DETECTION_INTERVAL_MS - elapsed);
                sleep(sleepTime);
            }
        }, "detection-thread");

        Timer uiTimer = new Timer(DISPLAY_INTERVAL_MS, e -> {
            BufferedImage frame = latestFrame.get();

            if (frame != null) {
                panel.setData(frame, latestDetections.get());
            }
        });

        captureThread.setDaemon(true);
        detectionThread.setDaemon(true);

        captureThread.start();
        detectionThread.start();
        uiTimer.start();

        while (running.get()) {
            sleep(300);
        }

        uiTimer.stop();

        try {
            session.close();
        } catch (Exception ignored) {
        }

        SwingUtilities.invokeLater(window::dispose);

        System.out.println("Stopped.");
    }

    private static FFmpegFrameGrabber createLowLatencyGrabber(String streamUrl) {
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(streamUrl);

        /*
         * Настройки против лишней задержки.
         * Для YouTube HLS они не уберут задержку самого YouTube,
         * но уменьшат задержку внутри FFmpeg/Java.
         */
        grabber.setOption("fflags", "nobuffer");
        grabber.setOption("flags", "low_delay");
        grabber.setOption("avioflags", "direct");
        grabber.setOption("max_delay", "0");
        grabber.setOption("probesize", "32768");
        grabber.setOption("analyzeduration", "0");

        grabber.setOption("reconnect", "1");
        grabber.setOption("reconnect_streamed", "1");
        grabber.setOption("reconnect_delay_max", "2");

        return grabber;
    }

    private static String getYouTubeStreamUrl(String youtubeUrl) throws Exception {
        String ytDlpPath = Files.exists(Path.of("/opt/homebrew/bin/yt-dlp"))
                ? "/opt/homebrew/bin/yt-dlp"
                : "yt-dlp";

        /*
         * Берём лёгкий video-only stream.
         * Не 720p 60 FPS, а до 360p и до 30 FPS.
         */
        ProcessBuilder pb = new ProcessBuilder(
                ytDlpPath,
                "--no-warnings",
                "-f",
                "bv*[height<=360][fps<=30]/best[height<=360][fps<=30]/worst[height<=360]/worst",
                "-g",
                youtubeUrl
        );

        pb.redirectErrorStream(true);

        Process process = pb.start();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream())
        )) {
            String line;

            while ((line = reader.readLine()) != null) {
                if (line.startsWith("http")) {
                    return line;
                }
            }
        }

        int exitCode = process.waitFor();
        throw new RuntimeException("Could not get YouTube stream URL. Exit code: " + exitCode);
    }

    private static List<Detection> detectBicycles(
            BufferedImage originalImage,
            OrtEnvironment env,
            OrtSession session,
            String inputName
    ) throws Exception {

        int originalWidth = originalImage.getWidth();
        int originalHeight = originalImage.getHeight();

        BufferedImage resized = resizeImage(originalImage, INPUT_SIZE, INPUT_SIZE);
        float[] inputData = imageToCHWFloatArrayFast(resized);

        try (OnnxTensor tensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(inputData),
                new long[]{1, 3, INPUT_SIZE, INPUT_SIZE}
        )) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(inputName, tensor);

            try (OrtSession.Result result = session.run(inputs)) {
                float[][][] output = (float[][][]) result.get(0).getValue();

                // YOLO11 output: [1][84][8400]
                float[][] predictions = output[0];

                List<Detection> candidates = new ArrayList<>();
                int predictionCount = predictions[0].length;

                for (int i = 0; i < predictionCount; i++) {
                    float centerX = predictions[0][i];
                    float centerY = predictions[1][i];
                    float width = predictions[2][i];
                    float height = predictions[3][i];

                    float bestScore = 0f;
                    int bestClassId = -1;

                    for (int classIndex = 4; classIndex < 84; classIndex++) {
                        float score = predictions[classIndex][i];

                        if (score > bestScore) {
                            bestScore = score;
                            bestClassId = classIndex - 4;
                        }
                    }

                    if (bestClassId == BICYCLE_CLASS_ID && bestScore >= CONF_THRESHOLD) {
                        float scaleX = originalWidth / (float) INPUT_SIZE;
                        float scaleY = originalHeight / (float) INPUT_SIZE;

                        int x1 = Math.round((centerX - width / 2f) * scaleX);
                        int y1 = Math.round((centerY - height / 2f) * scaleY);
                        int x2 = Math.round((centerX + width / 2f) * scaleX);
                        int y2 = Math.round((centerY + height / 2f) * scaleY);

                        x1 = clamp(x1, 0, originalWidth - 1);
                        y1 = clamp(y1, 0, originalHeight - 1);
                        x2 = clamp(x2, 0, originalWidth - 1);
                        y2 = clamp(y2, 0, originalHeight - 1);

                        candidates.add(new Detection(x1, y1, x2, y2, bestScore));
                    }
                }

                return nms(candidates, NMS_THRESHOLD);
            }
        }
    }

    private static BufferedImage resizeImage(BufferedImage source, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();

        g.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR
        );

        g.drawImage(source, 0, 0, width, height, null);
        g.dispose();

        return resized;
    }

    private static float[] imageToCHWFloatArrayFast(BufferedImage image) {
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        image.getRGB(0, 0, INPUT_SIZE, INPUT_SIZE, pixels, 0, INPUT_SIZE);

        float[] data = new float[3 * INPUT_SIZE * INPUT_SIZE];
        int channelSize = INPUT_SIZE * INPUT_SIZE;

        for (int i = 0; i < pixels.length; i++) {
            int rgb = pixels[i];

            int r = (rgb >> 16) & 0xFF;
            int g = (rgb >> 8) & 0xFF;
            int b = rgb & 0xFF;

            data[i] = r / 255.0f;
            data[channelSize + i] = g / 255.0f;
            data[2 * channelSize + i] = b / 255.0f;
        }

        return data;
    }

    private static List<Detection> nms(List<Detection> detections, float threshold) {
        if (detections.isEmpty()) {
            return Collections.emptyList();
        }

        detections.sort(Comparator.comparingDouble((Detection d) -> d.confidence).reversed());

        List<Detection> result = new ArrayList<>();
        boolean[] removed = new boolean[detections.size()];

        for (int i = 0; i < detections.size(); i++) {
            if (removed[i]) {
                continue;
            }

            Detection current = detections.get(i);
            result.add(current);

            for (int j = i + 1; j < detections.size(); j++) {
                if (removed[j]) {
                    continue;
                }

                Detection other = detections.get(j);

                if (iou(current, other) > threshold) {
                    removed[j] = true;
                }
            }
        }

        return result;
    }

    private static float iou(Detection a, Detection b) {
        int x1 = Math.max(a.x1, b.x1);
        int y1 = Math.max(a.y1, b.y1);
        int x2 = Math.min(a.x2, b.x2);
        int y2 = Math.min(a.y2, b.y2);

        int intersectionWidth = Math.max(0, x2 - x1);
        int intersectionHeight = Math.max(0, y2 - y1);

        int intersectionArea = intersectionWidth * intersectionHeight;

        int areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
        int areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

        int unionArea = areaA + areaB - intersectionArea;

        if (unionArea <= 0) {
            return 0f;
        }

        return intersectionArea / (float) unionArea;
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static class Detection {
        final int x1;
        final int y1;
        final int x2;
        final int y2;
        final float confidence;

        Detection(int x1, int y1, int x2, int y2, float confidence) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.confidence = confidence;
        }
    }

    private static class ImagePanel extends JPanel {
        private volatile BufferedImage image;
        private volatile List<Detection> detections = Collections.emptyList();

        ImagePanel() {
            setPreferredSize(new Dimension(1000, 700));
            setBackground(Color.BLACK);
        }

        void setData(BufferedImage image, List<Detection> detections) {
            this.image = image;
            this.detections = detections == null ? Collections.emptyList() : detections;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics graphics) {
            super.paintComponent(graphics);

            BufferedImage currentImage = image;

            if (currentImage == null) {
                return;
            }

            Graphics2D g = (Graphics2D) graphics;

            g.setRenderingHint(
                    RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR
            );

            int panelWidth = getWidth();
            int panelHeight = getHeight();

            double scale = Math.min(
                    panelWidth / (double) currentImage.getWidth(),
                    panelHeight / (double) currentImage.getHeight()
            );

            int drawWidth = (int) (currentImage.getWidth() * scale);
            int drawHeight = (int) (currentImage.getHeight() * scale);

            int offsetX = (panelWidth - drawWidth) / 2;
            int offsetY = (panelHeight - drawHeight) / 2;

            g.drawImage(currentImage, offsetX, offsetY, drawWidth, drawHeight, null);

            drawDetections(g, detections, scale, offsetX, offsetY);
        }

        private void drawDetections(
                Graphics2D g,
                List<Detection> detections,
                double scale,
                int offsetX,
                int offsetY
        ) {
            if (detections == null || detections.isEmpty()) {
                return;
            }

            g.setStroke(new BasicStroke(3));
            g.setColor(Color.GREEN);
            g.setFont(new Font("Arial", Font.BOLD, 18));

            for (Detection d : detections) {
                int x = offsetX + (int) (d.x1 * scale);
                int y = offsetY + (int) (d.y1 * scale);
                int w = (int) ((d.x2 - d.x1) * scale);
                int h = (int) ((d.y2 - d.y1) * scale);

                g.drawRect(x, y, w, h);

                String text = "Bicycle " + String.format("%.2f", d.confidence);
                g.drawString(text, x, Math.max(y - 8, 20));
            }
        }
    }
}