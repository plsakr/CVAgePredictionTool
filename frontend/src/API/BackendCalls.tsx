import io from 'socket.io-client'
import {ModelInfo} from "./MyTypes";
const backend = "http://127.0.0.1:5000";


// const socket = io(backend)
// let isConnected = false;

// console.log('tried doing things!')
// socket.on('connect', () => {
//     isConnected = true
//     console.log('SOCKET CONNECTED!')
// });

function predict(body: { image: string | ArrayBuffer }) {
    // if (isConnected) {
    //     socket.emit('predict', body)
    // }
}

function subscribeToInfo(callback: (data: ModelInfo) => void) {
    // console.log(`socket is connected: ${socket.connected}`)
    // if (isConnected) {
    //     socket.on('info', (data: ModelInfo) => callback(data))
    //     socket.emit('request-info')
    // } else {
    //     setTimeout(() => {
    //         subscribeToInfo(callback)
    //     }, 1000)
    // }

}

function subscribeToPrediction(callback: (label: string) => void) {
    // if (isConnected) {
    //     socket.on('predict-result', (data: string) => callback(data))
    // } else {
    //     setTimeout(() => {
    //         subscribeToPrediction(callback)
    //     }, 1000)
    // }
}

export {subscribeToInfo, subscribeToPrediction, predict};
