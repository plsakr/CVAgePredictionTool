"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.subscribeToInfo = void 0;
var socket_io_client_1 = require("socket.io-client");
var backend = "http://127.0.0.1:5000";
var socket = socket_io_client_1.io(backend);
function subscribeToInfo(callback) {
    socket.on('info', function (data) { return callback(data); });
    socket.emit('request-info');
}
exports.subscribeToInfo = subscribeToInfo;
