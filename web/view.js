/**
 * CredicelTestlifeSDK v5.0 - MediaPipe + FaceTec-style UI
 */
class CredicelTestlifeSDK {
    constructor(opts) {
        if (!opts) opts = {};
        this.apiKey = opts.ctlApiKey;
        this.container = opts.containerElement;
        this.transactionId = opts.transactionId;
        this.biometricId = opts.biometricIdentifier;
        this.comparativeDoc = opts.comparativeDoc;
        this.tokenCredicel = opts.tokenCredicel || null;
        this.primaryColor = opts.primaryColor || "#5f2978";
        this.accentColor = opts.secundaryColor || "#00d4ff";
        this.successCallback = opts.successCallback || function(){};
        this.sendDataCallback = opts.sendDataCallback || function(){};
        this.errorCallback = opts.errorCallback || function(){};
        this._video = null;
        this._canvas = null;
        this._ctx = null;
        this._statusText = "";
        this._subText = "";
        this._loader = null;
        this._step = "idle";
        this._interval = null;
        this._stableCount = 0;
        this._animFrame = null;
        this._detecting = false;
        this._isLowEnd = false;
        this._recorder = null;
        this._chunks = [];
        this._recording = false;
        this._ovalColor = "rgba(255,255,255,0.5)";
        this._ovalGlow = 0;
        this._ovalScale = 1.0;
        this._ovalTargetScale = 1.0;
        this._faceLandmarker = null;
        this._lastDetectTime = 0;
        this._epToken = "https://sdk.microtecnologiasmoviles.mx/fakeMatch/createDataFaceMatchSDK";
        this._epLiveness = "http://158.69.201.108:5006/v1/liveness";
        this._ineFile = null; // Optional INE image file
        this._gps = {};
        this._token = "";
        this._zoomReached = false;
        this._initialDist = 0;
        this._recStartTime = 0;
        this._retry = false;
        this._zoomReachedTime = 0;
        this._blinkCount = 0;
        this._eyesClosed = false;
        this._eyeCloseTime = 0;
        this._ldSamples = [];
        this._ldVariance = 0;
        this._challenge = null;
        this._challengeStartTime = 0;
        this._challengeRetries = 0;
        this._challengeCountdownInterval = null;
        this._injectCSS();
        this._buildLoader();
        // Liberar recursos al recargar pagina (critico en Android low-end)
        var self = this;
        window.addEventListener("beforeunload", function() { self._cleanup(); });
    }

    setIneImage(file) {
        this._ineFile = file || null;
    }

    async bioTestlifeProcess() {
        try { this._gps = await this._getGPS(); }
        catch (e) { this._gps = { lat: 0, lng: 0 }; }
        this._showLoader("Preparando...");
        this._isLowEnd = this._detectLowEnd();
        await this._loadMediaPipe();
        this._token = this.tokenCredicel || await this._genToken();
        this._buildUI();
        this._startCamera();
    }

    presentErrorFeedback(code) {
        code = code || "";
        var old = document.querySelector(".ctl-err");
        if (old) old.remove();
        var d = document.createElement("div");
        d.className = "ctl-err";
        d.innerHTML = '<div class="ctl-err-box">' +
            '<div class="ctl-err-x">\u2715</div>' +
            '<h3>Verificacion fallida</h3>' +
            '<p>Asegurate de:</p>' +
            '<ul><li>Sin lentes ni accesorios</li><li>Buena iluminacion</li>' +
            '<li>Rostro de frente</li><li>Sin pantallas de fondo</li></ul>' +
            '<button onclick="location.reload()">Reintentar</button>' +
            (code ? '<span style="color:#666;font-size:11px">' + code + '</span>' : '') +
            '</div>';
        document.body.appendChild(d);
        this._cleanup();
    }

    getRecordedVideoBlob() {
        return this._chunks.length ? new Blob(this._chunks, { type: "video/webm" }) : null;
    }

    downloadRecordedVideo(f) {
        f = f || "video.webm";
        var b = this.getRecordedVideoBlob(); if (!b) return;
        var a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = f; a.click();
    }

    _buildUI() {
        document.body.style.cssText = "margin:0;padding:0;overflow:hidden;background:#000;";
        this.container.style.cssText = "position:fixed;inset:0;z-index:100;background:#000;";
        this.container.innerHTML =
            '<video id="ctl-vid" playsinline autoplay muted style="display:none"></video>' +
            '<canvas id="ctl-cvs" style="position:absolute;inset:0;width:100%;height:100%;z-index:1"></canvas>' +
            '<div id="ctl-card" style="position:absolute;bottom:72px;left:50%;transform:translateX(-50%);z-index:2;' +
            'background:rgba(0,0,0,0.58);border:1px solid rgba(255,255,255,0.13);' +
            'border-radius:20px;padding:12px 26px;text-align:center;white-space:nowrap;' +
            'pointer-events:none;opacity:0;transition:opacity 0.3s ease">' +
            '<div id="ctl-card-main" style="font-size:15px;font-weight:600;color:#fff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif"></div>' +
            '<div id="ctl-card-sub" style="font-size:12px;font-weight:400;color:rgba(255,255,255,0.5);margin-top:3px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif"></div>' +
            '</div>';
        this._video = document.getElementById("ctl-vid");
        this._canvas = document.getElementById("ctl-cvs");
        this._ctx = this._canvas.getContext("2d", { willReadFrequently: false, alpha: false });
        this._cardEl = document.getElementById("ctl-card");
        this._cardMain = document.getElementById("ctl-card-main");
        this._cardSub = document.getElementById("ctl-card-sub");
        this._statusText = "";
        this._subText = "";
    }

    _injectCSS() {
        if (document.getElementById("ctl-css")) return;
        var s = document.createElement("style");
        s.id = "ctl-css";
        var pc = this.primaryColor, ac = this.accentColor;
        s.textContent =
            '#ctl-loader{position:fixed;inset:0;background:rgba(0,0,0,0.9);display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:9999;gap:14px}' +
            '#ctl-loader.hide{display:none}' +
            '.ctl-spin{width:40px;height:40px;border:3px solid rgba(255,255,255,0.1);border-top-color:' + ac + ';border-radius:50%;animation:ctl-sp .7s linear infinite}' +
            '@keyframes ctl-sp{to{transform:rotate(360deg)}}' +
            '#ctl-loader span{color:#fff;font-size:14px;font-weight:500;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif}' +
            '#ctl-loader button{padding:10px 28px;border:none;border-radius:20px;background:' + pc + ';color:#fff;font-weight:700;font-size:14px;cursor:pointer}' +
            '#ctl-loader button.hide{display:none}' +
            '.ctl-err{position:fixed;inset:0;background:rgba(0,0,0,0.95);display:flex;align-items:center;justify-content:center;z-index:10000;padding:20px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif}' +
            '.ctl-err-box{background:#1a1a1a;border-radius:20px;max-width:340px;width:100%;padding:32px 24px;text-align:center;color:#fff}' +
            '.ctl-err-x{width:56px;height:56px;border-radius:50%;background:#ef4444;color:#fff;font-size:28px;font-weight:700;display:flex;align-items:center;justify-content:center;margin:0 auto 16px}' +
            '.ctl-err-box h3{margin:0 0 6px;font-size:17px}' +
            '.ctl-err-box p{color:#999;font-size:13px;margin:0 0 12px}' +
            '.ctl-err-box ul{text-align:left;color:#bbb;font-size:13px;padding-left:18px;margin:0 0 20px;line-height:1.9}' +
            '.ctl-err-box button{width:100%;padding:14px;border:none;border-radius:14px;background:' + pc + ';color:#fff;font-weight:700;font-size:15px;cursor:pointer}';
        document.head.appendChild(s);
    }

    async _startCamera() {
        var self = this;
        var isAndroid = /android/i.test(navigator.userAgent);
        try {
            // Forzar liberacion de cualquier stream previo (critico en Android)
            if (window.stream) {
                window.stream.getTracks().forEach(function(t) { t.stop(); });
                window.stream = null;
            }
            if (this._video && this._video.srcObject) {
                this._video.pause();
                this._video.srcObject = null;
            }
            // Android necesita tiempo para liberar el hardware de la camara
            if (isAndroid) {
                await new Promise(function(r) { setTimeout(r, 500); });
            }

            var w = this._isLowEnd ? 640 : 1280;
            var h = this._isLowEnd ? 480 : 720;
            var stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "user", width: { ideal: w }, height: { ideal: h } }, audio: false
                });
            } catch (e1) {
                stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
            }
            window.stream = stream;

            this._video.srcObject = stream;
            try { await this._video.play(); } catch (x) {}
            await new Promise(function(resolve) {
                var check = function() {
                    if (self._video.videoWidth > 0 && self._video.videoHeight > 0) return resolve();
                    setTimeout(check, 100);
                };
                check();
                setTimeout(resolve, 5000);
            });

            // Esperar a que la camara tenga brillo suficiente (Android low-end)
            if (isAndroid) {
                await new Promise(function(resolve) {
                    var tempCanvas = document.createElement("canvas");
                    tempCanvas.width = 160;
                    tempCanvas.height = 120;
                    var tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });
                    var attempts = 0;
                    var checkBrightness = function() {
                        attempts++;
                        if (attempts > 50) return resolve(); // Max 5 segundos
                        try {
                            tempCtx.drawImage(self._video, 0, 0, 160, 120);
                            var imgData = tempCtx.getImageData(40, 30, 80, 60);
                            var pixels = imgData.data;
                            var sum = 0;
                            for (var i = 0; i < pixels.length; i += 4) {
                                sum += pixels[i] + pixels[i+1] + pixels[i+2];
                            }
                            var avgBrightness = sum / (pixels.length * 0.75);
                            console.log("[camera] brightness=" + avgBrightness.toFixed(1) + " attempt=" + attempts);
                            if (avgBrightness > 30) {
                                console.log("[camera] brightness OK, starting render");
                                return resolve();
                            }
                        } catch (e) {}
                        setTimeout(checkBrightness, 100);
                    };
                    setTimeout(checkBrightness, 200);
                });
            }

            this._canvas.width = window.innerWidth;
            this._canvas.height = window.innerHeight;

            // Warm-up: primera inferencia de MediaPipe (compila shaders/WASM)
            // Se hace antes de ocultar el loader para que el lag no se vea
            try {
                this._faceLandmarker.detectForVideo(this._video, performance.now());
            } catch (e) {}

            console.log("[camera] video=" + this._video.videoWidth + "x" + this._video.videoHeight +
                " canvas=" + this._canvas.width + "x" + this._canvas.height +
                " lowEnd=" + this._isLowEnd + " android=" + isAndroid);
            this._hideLoader();
            this._step = "detecting";
            this._stableCount = 0;
            this._startRender();
            this._startDetection();

            // Manejar bloqueo/desbloqueo de pantalla en Android
            if (!this._visHandler) {
                this._visHandler = function() {
                    if (document.visibilityState === "visible" && self._video && self._video.srcObject) {
                        try { self._video.play(); } catch (x) {}
                    }
                };
                document.addEventListener("visibilitychange", this._visHandler);
            }
        } catch (e) {
            this._showLoader("No se pudo acceder a la camara", true);
        }
    }

    _startRender() {
        if (this._animFrame) cancelAnimationFrame(this._animFrame);
        var self = this, lastTime = 0;
        var interval = this._isLowEnd ? 33 : 16;
        var loop = function(time) {
            if (time - lastTime >= interval) { self._renderFrame(); lastTime = time; }
            self._animFrame = requestAnimationFrame(loop);
        };
        this._animFrame = requestAnimationFrame(loop);
    }

    _stopRender() {
        if (this._animFrame) { cancelAnimationFrame(this._animFrame); this._animFrame = null; }
    }

    _renderFrame() {
        var ctx = this._ctx;
        if (!ctx) return;
        var cw = this._canvas.width, ch = this._canvas.height;
        if (!cw || !ch) return;
        var vw = this._video.videoWidth, vh = this._video.videoHeight;
        if (!vw || !vh) return;

        // Asegurar estado limpio del contexto
        ctx.globalAlpha = 1.0;
        ctx.globalCompositeOperation = "source-over";
        ctx.clearRect(0, 0, cw, ch);

        var canvasRatio = cw / ch, videoRatio = vw / vh;
        var sx = 0, sy = 0, sw = vw, sh = vh;
        if (videoRatio > canvasRatio) { sw = vh * canvasRatio; sx = (vw - sw) / 2; }
        else { sh = vw / canvasRatio; sy = (vh - sh) / 2; }
        ctx.save(); ctx.translate(cw, 0); ctx.scale(-1, 1);
        ctx.drawImage(this._video, sx, sy, sw, sh, 0, 0, cw, ch); ctx.restore();
        var cx = cw / 2, cy = ch * 0.40;
        if (Math.abs(this._ovalScale - this._ovalTargetScale) > 0.005) {
            this._ovalScale += (this._ovalTargetScale - this._ovalScale) * 0.12;
        } else { this._ovalScale = this._ovalTargetScale; }
        var rx = cw * 0.34 * this._ovalScale, ry = rx * 1.30;
        // Mascara oscura fuera del ovalo (FaceTime style)
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, cw, ch);
        ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2, true);
        ctx.fillStyle = "rgba(0,0,0,0.72)"; ctx.fill("evenodd"); ctx.restore();
        // Borde del ovalo
        ctx.save(); ctx.beginPath(); ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        ctx.strokeStyle = this._ovalColor; ctx.lineWidth = 3.5;
        if (this._ovalGlow > 0) { ctx.shadowBlur = this._ovalGlow; ctx.shadowColor = this._ovalColor; }
        ctx.stroke(); ctx.restore();
        // Arco de progreso cuando el zoom fue alcanzado
        if (this._step === "zoomin" && this._zoomReached && this._zoomReachedTime > 0) {
            var prog = Math.min(1, (Date.now() - this._zoomReachedTime) / 3000);
            ctx.save();
            ctx.beginPath();
            ctx.ellipse(cx, cy, rx + 6, ry + 6, 0, -Math.PI / 2, -Math.PI / 2 + prog * Math.PI * 2);
            ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 4; ctx.lineCap = "round";
            ctx.shadowBlur = 12; ctx.shadowColor = "#22c55e";
            ctx.stroke(); ctx.restore();
        }
    }

    _setOval(color, glow, scale) {
        this._ovalColor = color;
        this._ovalGlow = glow || 0;
        this._ovalTargetScale = (scale !== undefined) ? scale : 1.0;
    }

    _setStatus(main, sub) {
        this._statusText = main; this._subText = sub || "";
        if (this._cardEl) {
            if (main) {
                this._cardMain.textContent = main;
                this._cardSub.textContent = sub || "";
                this._cardSub.style.display = sub ? "block" : "none";
                this._cardEl.style.opacity = "1";
            } else {
                this._cardEl.style.opacity = "0";
            }
        }
    }

    _startDetection() {
        if (this._interval) clearInterval(this._interval);
        var detInterval = this._isLowEnd ? 250 : 130;
        var self = this;
        this._interval = setInterval(function() {
            if (self._step === "uploading" || self._step === "done" || self._step === "finishing") return;
            if (self._detecting) return;
            self._detecting = true;
            try {
                var now = performance.now();
                if (now <= self._lastDetectTime) { self._detecting = false; return; }
                self._lastDetectTime = now;
                var results = self._faceLandmarker.detectForVideo(self._video, now);
                if (!results.faceLandmarks || !results.faceLandmarks.length) {
                    self._stableCount = 0;
                    if (self._step === "detecting") {
                        self._setOval("rgba(255,255,255,0.4)", 0, 1.0);
                        self._setStatus("Posiciona tu rostro en el ovalo", "");
                    }
                    self._detecting = false; return;
                }
                var landmarks = results.faceLandmarks[0];
                var faceTop = landmarks[10], faceBottom = landmarks[152];
                var faceHeight = Math.abs(faceBottom.y - faceTop.y);
                var dist = 1.0 / faceHeight;

                // MediaPipe dist: ~6+ muy lejos, ~3-5 normal, ~2 cerca, <1.5 pegado
                if (self._step === "detecting") {
                    if (dist >= 5.5) {
                        self._setOval("rgba(255,255,255,0.4)", 0, 0.85);
                        self._setStatus("Acercate un poco mas", "");
                        self._stableCount = 0; self._detecting = false; return;
                    }
                    if (dist <= 1.8) {
                        self._setOval("#ff6600", 8, 1.15);
                        self._setStatus("Alejate un poco", "");
                        self._stableCount = 0; self._detecting = false; return;
                    }
                    // Rango ideal para detectar: dist entre 2.5 y 5.5
                    if (dist > 5.0) {
                        self._setOval("rgba(255,255,255,0.4)", 0, 0.9);
                        self._setStatus("Acercate un poco", "");
                        self._stableCount = 0; self._detecting = false; return;
                    }
                    self._stableCount++;
                    if (self._stableCount > 12) {
                        self._step = "challenge";
                        self._challengeRetries = 0;
                        self._setOval("#22c55e", 15, 1.0);
                        self._setStatus("Rostro detectado", "");
                        setTimeout(function() {
                            if (self._step === "challenge") self._startChallenge();
                        }, 600);
                    } else {
                        self._setOval(self.accentColor, 10, 1.0);
                        self._setStatus("Mantente asi...", "Verificando");
                    }
                    self._detecting = false; return;
                }

                if (self._step === "challenge") {
                    self._checkChallenge(results, landmarks);
                    self._detecting = false; return;
                }

                if (self._step === "zoomin") {
                    self._detectBlink(results);
                    self._sampleLandmarkDist(landmarks);
                    var elapsed = Date.now() - self._recStartTime;
                    var targetDist = self._initialDist * 0.70;
                    console.log("[zoom] dist=" + dist.toFixed(2) + " initial=" + self._initialDist.toFixed(2) + " target=" + targetDist.toFixed(2) + " reached=" + self._zoomReached);

                    if (dist <= targetDist && !self._zoomReached) {
                        self._zoomReached = true;
                        self._zoomReachedTime = Date.now();
                        self._setOval("#22c55e", 18, 1.35);
                        self._setStatus("Perfecto, mantente asi", "");
                    }
                    if (self._zoomReached && (Date.now() - self._zoomReachedTime) >= 3000) {
                        self._step = "finishing";
                        self._setOval("#22c55e", 18, 1.35);
                        self._setStatus("Captura completa", "");
                        setTimeout(function() { self._stopRec(); }, 200);
                        self._detecting = false; return;
                    }
                    if (!self._zoomReached && elapsed >= 8000) {
                        self._abortRec(); self._step = "idle";
                        self._stopRender(); self._cleanup();
                        self.presentErrorFeedback("DEPTH_TIMEOUT");
                        self._detecting = false; return;
                    }
                    if (dist > 7.0) {
                        self._abortRec(); self._step = "detecting";
                        self._stableCount = 0;
                        self._zoomReached = false; self._zoomReachedTime = 0;
                        self._setOval("rgba(255,255,255,0.4)", 0, 1.0);
                        self._setStatus("Posiciona tu rostro en el ovalo", "");
                        self._detecting = false; return;
                    }
                    if (!self._zoomReached && elapsed > 600) {
                        self._setOval(self.accentColor, 12, 1.35);
                        self._setStatus("Acerca tu rostro", "Lentamente hacia el ovalo");
                    }
                    self._detecting = false; return;
                }
            } catch (ex) {
                console.warn("Detection error:", ex);
            }
            self._detecting = false;
        }, detInterval);
    }

    _startChallenge() {
        var pool = [
            { id: "blink",     main: "Parpadea",            sub: "Cierra y abre los ojos" },
            { id: "turnLeft",  main: "Gira a la izquierda", sub: "Voltea tu cabeza lentamente" },
            { id: "turnRight", main: "Gira a la derecha",   sub: "Voltea tu cabeza lentamente" },
            { id: "smile",     main: "Sonrie",              sub: "Muestra una sonrisa" }
        ];
        this._challenge = pool[Math.floor(Math.random() * pool.length)];
        this._challengeStartTime = Date.now();
        this._eyesClosed = false;
        this._eyeCloseTime = 0;
        this._setOval(this.accentColor, 12, 1.0);
        this._setStatus(this._challenge.main, this._challenge.sub);
        var self = this;
        var secs = 5;
        if (this._challengeCountdownInterval) clearInterval(this._challengeCountdownInterval);
        this._challengeCountdownInterval = setInterval(function() {
            secs--;
            if (secs > 0 && self._step === "challenge") {
                self._setStatus(self._challenge.main, self._challenge.sub + " (" + secs + "s)");
            }
        }, 1000);
    }

    _checkChallenge(results, landmarks) {
        var c = this._challenge;
        if (!c) return;
        var elapsed = Date.now() - this._challengeStartTime;
        var passed = false;
        var cats = (results.faceBlendshapes && results.faceBlendshapes.length)
            ? results.faceBlendshapes[0].categories : [];

        if (c.id === "blink") {
            var blLeft = 0, blRight = 0;
            for (var i = 0; i < cats.length; i++) {
                if (cats[i].categoryName === "eyeBlinkLeft")  blLeft  = cats[i].score;
                if (cats[i].categoryName === "eyeBlinkRight") blRight = cats[i].score;
            }
            var bothClosed = blLeft > 0.4 && blRight > 0.4;
            var now = Date.now();
            if (bothClosed && !this._eyesClosed) {
                this._eyesClosed = true; this._eyeCloseTime = now;
            } else if (!bothClosed && this._eyesClosed) {
                var dur = now - this._eyeCloseTime;
                if (dur >= 80 && dur <= 450) passed = true;
                this._eyesClosed = false;
            }
        } else if (c.id === "turnLeft" || c.id === "turnRight") {
            var fw = Math.abs(landmarks[454].x - landmarks[234].x);
            if (fw > 0) {
                var offset = (landmarks[1].x - (landmarks[234].x + landmarks[454].x) / 2) / fw;
                if (c.id === "turnLeft"  && offset >  0.18) passed = true;
                if (c.id === "turnRight" && offset < -0.18) passed = true;
            }
        } else if (c.id === "smile") {
            var smL = 0, smR = 0;
            for (var j = 0; j < cats.length; j++) {
                if (cats[j].categoryName === "mouthSmileLeft")  smL = cats[j].score;
                if (cats[j].categoryName === "mouthSmileRight") smR = cats[j].score;
            }
            if ((smL + smR) / 2 > 0.5) passed = true;
        }

        if (passed) {
            clearInterval(this._challengeCountdownInterval);
            this._challengeCountdownInterval = null;
            this._challenge = null;
            var fh = Math.abs(landmarks[152].y - landmarks[10].y);
            this._step = "zoomin";
            this._zoomReached = false;
            this._zoomReachedTime = 0;
            this._initialDist = fh > 0 ? 1.0 / fh : 3.0;
            this._blinkCount = 0;
            this._eyesClosed = false;
            this._eyeCloseTime = 0;
            this._ldSamples = [];
            this._recStartTime = Date.now();
            this._setOval("#22c55e", 18, 1.0);
            this._setStatus("Perfecto", "");
            var self = this;
            setTimeout(function() {
                if (self._step === "zoomin") {
                    self._setOval(self.accentColor, 12, 1.35);
                    self._setStatus("Acerca tu rostro", "Lentamente hacia el ovalo");
                    self._startRec();
                }
            }, 600);
            return;
        }

        if (elapsed >= 5000) {
            clearInterval(this._challengeCountdownInterval);
            this._challengeCountdownInterval = null;
            this._challengeRetries++;
            if (this._challengeRetries >= 2) {
                this._step = "idle";
                this._stopRender();
                this._cleanup();
                this.presentErrorFeedback("CHALLENGE_TIMEOUT");
                return;
            }
            this._startChallenge();
        }
    }

    _sampleLandmarkDist(lm) {
        // 5 pares de landmarks con geometria interna — invariantes a movimiento rigido
        // Boca ancho, apertura labial, ojo izq ancho, ojo der ancho, nariz-labio
        var pairs = [[61,291],[13,14],[33,133],[263,362],[1,13]];
        var sample = [];
        for (var i = 0; i < pairs.length; i++) {
            var a = lm[pairs[i][0]], b = lm[pairs[i][1]];
            var dx = a.x - b.x, dy = a.y - b.y;
            sample.push(Math.sqrt(dx*dx + dy*dy));
        }
        this._ldSamples.push(sample);
    }

    _calcLandmarkVariance() {
        var n = this._ldSamples.length;
        if (n < 5) return 0;
        var numPairs = this._ldSamples[0].length;
        var totalStd = 0;
        for (var p = 0; p < numPairs; p++) {
            var sum = 0;
            for (var i = 0; i < n; i++) sum += this._ldSamples[i][p];
            var mean = sum / n;
            var sq = 0;
            for (var i = 0; i < n; i++) { var d = this._ldSamples[i][p] - mean; sq += d*d; }
            totalStd += Math.sqrt(sq / n);
        }
        return totalStd / numPairs;
    }

    _detectBlink(results) {
        try {
            if (!results.faceBlendshapes || !results.faceBlendshapes.length) return;
            var cats = results.faceBlendshapes[0].categories;
            var left = 0, right = 0;
            for (var i = 0; i < cats.length; i++) {
                if (cats[i].categoryName === "eyeBlinkLeft") left = cats[i].score;
                else if (cats[i].categoryName === "eyeBlinkRight") right = cats[i].score;
            }
            var bothClosed = left > 0.4 && right > 0.4;
            var now = Date.now();
            if (bothClosed && !this._eyesClosed) {
                this._eyesClosed = true;
                this._eyeCloseTime = now;
            } else if (!bothClosed && this._eyesClosed) {
                var dur = now - this._eyeCloseTime;
                if (dur >= 80 && dur <= 450) {
                    this._blinkCount++;
                    console.log("[blink] count=" + this._blinkCount + " dur=" + dur + "ms");
                }
                this._eyesClosed = false;
            }
        } catch (e) {}
    }

    _startRec() {
        this._chunks = [];
        var stream = this._video.srcObject;
        var mime = "video/webm;codecs=vp9";
        if (!MediaRecorder.isTypeSupported(mime)) {
            mime = "video/webm;codecs=vp8";
            if (!MediaRecorder.isTypeSupported(mime)) {
                mime = "video/webm";
                if (!MediaRecorder.isTypeSupported(mime)) mime = "video/mp4";
            }
        }
        try { this._recorder = new MediaRecorder(stream, { mimeType: mime }); }
        catch (e) { this._showLoader("Navegador no soporta grabacion", true); return; }
        var self = this;
        this._recorder.ondataavailable = function(e) { if (e.data.size > 0) self._chunks.push(e.data); };
        this._recorder.onstop = function() { self._processVideo(); };
        this._recording = true;
        this._recorder.start(100);
    }

    _stopRec() {
        if (this._recorder && this._recorder.state !== "inactive") this._recorder.stop();
        this._recording = false;
        this._setOval("#22c55e", 18, 1.0);
        this._setStatus("Captura completa", "Procesando...");
    }

    _abortRec() {
        if (this._recorder && this._recorder.state !== "inactive") {
            this._recorder.onstop = null; this._recorder.stop();
        }
        this._recording = false; this._chunks = [];
    }

    async _processVideo() {
        this._step = "uploading";
        this._stopRender();
        this._ldVariance = this._calcLandmarkVariance();
        // DEBUG: veredicto temporal — REMOVER antes de produccion
        var blinkOk = this._blinkCount > 0;
        var varOk = this._ldVariance > 0.0015;
        // Solo bloquear en frontend si AMBAS señales fallan (foto estatica clara)
        // Si hay varianza pero no parpadeo, es ambiguo — el backend decide
        var isLive = varOk; // varianza minima requerida; blink va como metadata al backend
        var debugAccion = !varOk ? "NO SE SUBIRIA (foto estatica detectada)" :
                          (!blinkOk ? "SE SUBIRIA — sin parpadeo, backend decide" : "SE SUBIRIA AL SERVIDOR");
        alert(
            "[DEBUG LIVENESS]\n" +
            "Parpadeos: " + this._blinkCount + " → " + (blinkOk ? "OK" : "FALLO") + "\n" +
            "Varianza inter-landmark: " + this._ldVariance.toFixed(6) + " → " + (varOk ? "OK" : "FALLO") + "\n" +
            "─────────────────────\n" +
            "Veredicto: " + (isLive ? "PERSONA REAL" : "POSIBLE SPOOF") + "\n" +
            "Accion: " + debugAccion
        );
        if (!isLive) {
            this._showLoader("Verificacion fallida", true);
            this.errorCallback({ success: false, error: "FRONTEND_SPOOF_DETECTED" });
            return;
        }
        this._showLoader("Subiendo video...");
        this._cleanup();
        this.sendDataCallback();
        var blob = new Blob(this._chunks, { type: "video/webm" });
        await this._upload(blob, "testlife_" + Date.now() + ".webm");
    }

    async _upload(blob, name) {
        var fd = new FormData();
        fd.append("video", blob, name);
        if (this._ineFile) {
            fd.append("ine_image", this._ineFile, this._ineFile.name);
        }
        var ctrl = new AbortController();
        var to = setTimeout(function() { ctrl.abort(); }, 60000);
        try {
            var res = await fetch(this._epLiveness, { method: "POST", body: fd, signal: ctrl.signal });
            clearTimeout(to);
            var json = await res.json();
            this._step = "done"; this._hideLoader();
            this.container.style.cssText = ""; document.body.style.cssText = "";
            this.container.innerHTML = "";
            this.successCallback({
                success: true,
                result: json
            });
        } catch (err) {
            clearTimeout(to);
            if (!this._retry) { this._retry = true; return this._upload(blob, name); }
            alert(err);
            this._showLoader("No se pudo conectar con el servidor", true);
            this.errorCallback({ success: false, error: err.message });
        }
    }

    _cleanup() {
        if (this._interval) { clearInterval(this._interval); this._interval = null; }
        if (this._challengeCountdownInterval) { clearInterval(this._challengeCountdownInterval); this._challengeCountdownInterval = null; }
        // Liberar stream de forma agresiva para Android
        if (this._video) {
            this._video.pause();
            this._video.srcObject = null;
        }
        if (window.stream) {
            window.stream.getTracks().forEach(function(t) { t.stop(); });
            window.stream = null;
        }
        // Liberar MediaPipe FaceLandmarker (libera contexto WebGL/GPU)
        if (this._faceLandmarker) {
            try { this._faceLandmarker.close(); } catch (e) {}
            this._faceLandmarker = null;
        }
        if (this._visHandler) {
            document.removeEventListener("visibilitychange", this._visHandler);
            this._visHandler = null;
        }
        this._stopRender();
    }

    _buildLoader() {
        if (document.getElementById("ctl-loader")) return;
        var d = document.createElement("div");
        d.id = "ctl-loader"; d.className = "hide";
        d.innerHTML = '<div class="ctl-spin"></div><span id="ctl-lmsg">Cargando...</span>' +
            '<button id="ctl-lbtn" class="hide">Aceptar</button>';
        document.body.appendChild(d);
        this._loader = d;
        var self = this;
        document.getElementById("ctl-lbtn").onclick = function() { self._hideLoader(); self.errorCallback(); };
    }

    _showLoader(msg, btn) {
        if (!this._loader) this._buildLoader();
        this._loader.classList.remove("hide");
        document.getElementById("ctl-lmsg").textContent = msg || "Cargando...";
        document.getElementById("ctl-lbtn").classList.toggle("hide", !btn);
    }

    _hideLoader() { if (this._loader) this._loader.classList.add("hide"); }

    _detectLowEnd() {
        var cores = navigator.hardwareConcurrency || 2;
        var mem = navigator.deviceMemory || 4;
        if (cores <= 4 || mem <= 2) return true;
        if (/android/i.test(navigator.userAgent) && cores <= 4) return true;
        return false;
    }

    async _geoPerm() {
        if (!navigator.permissions) return "unknown";
        return (await navigator.permissions.query({ name: "geolocation" })).state;
    }

    async _getGPS() {
        return new Promise(function(ok, no) {
            if (!navigator.geolocation) return no("Geolocalizacion no soportada.");
            navigator.geolocation.getCurrentPosition(
                function(p) { ok({ lat: p.coords.latitude, lng: p.coords.longitude }); },
                function(e) { no(["", "Habilita la ubicacion.", "Ubicacion no disponible.", "Tiempo agotado."][e.code] || "Error GPS."); },
                { enableHighAccuracy: true, timeout: 8000, maximumAge: 0 }
            );
        });
    }

    async _genToken() {
        var fd = new FormData();
        fd.append("identifier", this.transactionId);
        fd.append("curp", this.biometricId);
        fd.append("codificado", this.comparativeDoc);
        fd.append("X-Credicel-Key", this.apiKey);
        try {
            var r = await fetch(this._epToken, { method: "POST", body: fd });
            var j = await r.json();
            return j.success ? j.tokenCredicel : null;
        } catch (e) { return null; }
    }

    async _loadMediaPipe() {
        var cdnBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
        if (!window.FilesetResolver) {
            await new Promise(function(ok, no) {
                var script = document.createElement("script");
                script.type = "module";
                script.textContent =
                    'import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";' +
                    'window.FilesetResolver = FilesetResolver;' +
                    'window.FaceLandmarker = FaceLandmarker;' +
                    'window.dispatchEvent(new Event("mediapipe-ready"));';
                document.head.appendChild(script);
                window.addEventListener("mediapipe-ready", ok, { once: true });
                setTimeout(function() { no("Timeout cargando MediaPipe"); }, 20000);
            });
        }
        // Reusar instancia existente si no fue cerrada
        if (this._faceLandmarker) return;
        var vision = await window.FilesetResolver.forVisionTasks(cdnBase);
        var opts = {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: false
        };
        try {
            this._faceLandmarker = await window.FaceLandmarker.createFromOptions(vision, opts);
            console.log("[mediapipe] delegate=GPU");
        } catch (gpuErr) {
            console.warn("[mediapipe] GPU failed, falling back to CPU:", gpuErr);
            opts.baseOptions.delegate = "CPU";
            this._faceLandmarker = await window.FaceLandmarker.createFromOptions(vision, opts);
            console.log("[mediapipe] delegate=CPU (fallback)");
        }
    }
}
