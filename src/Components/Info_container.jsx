function Info_container(props) {
  return (
    <>
      <div id='info-container'>
        <p>
          model: <span id='model-info'>{props.selectedModel}</span>
        </p>
        <select value={props.selectedModel} onChange={e => props.setSelectedModel(e.target.value)}>
          <option value="cocoSsd">Coco-ssd</option>
          <option value="yolov8n">yolov8n</option>
        </select>
      </div>
    </>
  )
}

export default Info_container