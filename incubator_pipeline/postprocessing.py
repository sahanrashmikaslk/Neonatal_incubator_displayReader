"""
Post-processing module for neonatal incubator display readings.

This module provides validation, correction, and smoothing logic for OCR predictions
from incubator display readings. It handles:
- Range validation
- Decimal correction
- Integer enforcement
- Confidence filtering
- Temporal smoothing for live video streams
"""

# Medical ranges for neonatal incubator vital signs
VALUE_RANGES = {
    'heart_rate_value': {
        'min': 60,
        'max': 220,
        'decimals': 0,
        'unit': 'bpm',
        'description': 'Heart Rate',
        'integer_only': True  # Heart rate is always an integer
    },
    'humidity_value': {
        'min': 30,
        'max': 95,
        'decimals': 0,
        'unit': '%',
        'description': 'Humidity',
        'integer_only': True  # Humidity is always an integer
    },
    'skin_temp_value': {
        'min': 32.0,
        'max': 39.0,
        'decimals': 1,
        'unit': '°C',
        'description': 'Skin Temperature',
        'integer_only': False  # Can have decimal (e.g., 36.5°C)
    },
    'spo2_value': {
        'min': 70,
        'max': 100,
        'decimals': 0,
        'unit': '%',
        'description': 'SpO2',
        'integer_only': True  # SpO2 is always an integer
    }
}

# List of numeric classes (for convenience)
NUMERIC_CLASSES = list(VALUE_RANGES.keys())


def try_fix_decimal(value_str, expected_decimals, min_val, max_val, integer_only=False):
    """
    Attempt to fix missing or misplaced decimal points.
    
    Args:
        value_str: Raw OCR string value
        expected_decimals: Expected number of decimal places
        min_val: Minimum valid value
        max_val: Maximum valid value
        integer_only: If True, removes any decimal points
        
    Returns:
        Corrected value as string, or None if unfixable
        
    Examples:
        >>> try_fix_decimal('356', 1, 32, 39, False)
        '35.6'
        >>> try_fix_decimal('145.5', 0, 60, 220, True)
        '145'
    """
    if not value_str or not value_str.replace('.', '').replace('%', '').isdigit():
        return None
    
    # Remove any non-numeric characters except decimal
    clean = value_str.replace('%', '').strip()
    
    # For integer-only values (heart rate, humidity, SpO2), remove decimals
    if integer_only:
        if '.' in clean:
            # Remove decimal point and everything after it
            clean = clean.split('.')[0]
        try:
            val = int(clean)
            if min_val <= val <= max_val:
                return str(val)
        except ValueError:
            pass
        return None
    
    # If already has correct decimal places and in range, return as-is
    try:
        val = float(clean)
        if min_val <= val <= max_val:
            return clean
    except ValueError:
        pass
    
    # Try to fix missing decimal point
    if expected_decimals > 0 and '.' not in clean:
        # Try inserting decimal at various positions
        for i in range(len(clean) - expected_decimals, 0, -1):
            candidate = clean[:i] + '.' + clean[i:]
            try:
                val = float(candidate)
                if min_val <= val <= max_val:
                    return candidate
            except ValueError:
                continue
    
    # Try removing extra zeros
    if clean.startswith('0') and len(clean) > 1:
        clean = clean.lstrip('0')
        try:
            val = float(clean)
            if min_val <= val <= max_val:
                return clean
        except ValueError:
            pass
    
    return None


def validate_and_correct_value(class_name, raw_value, ocr_confidence=0.0, min_confidence=0.5):
    """
    Validate OCR reading against expected ranges and apply corrections.
    
    Args:
        class_name: Name of the parameter class (e.g., 'heart_rate_value')
        raw_value: Raw OCR string value
        ocr_confidence: OCR confidence score (0-1)
        min_confidence: Minimum acceptable OCR confidence
        
    Returns:
        dict with keys:
            - 'valid': bool, whether value is valid
            - 'corrected_value': str or None, corrected value if valid
            - 'raw_value': str, original raw value
            - 'issues': list of issue descriptions
    """
    if class_name not in VALUE_RANGES:
        return {
            'valid': False,
            'corrected_value': None,
            'raw_value': raw_value,
            'issues': ['Unknown class']
        }
    
    config = VALUE_RANGES[class_name]
    issues = []
    
    # Check confidence
    if ocr_confidence < min_confidence:
        issues.append(f'Low OCR confidence: {ocr_confidence:.2f} < {min_confidence}')
    
    # Check if value exists
    if raw_value is None or raw_value == '':
        return {
            'valid': False,
            'corrected_value': None,
            'raw_value': raw_value,
            'issues': ['No value detected']
        }
    
    # Try to parse as number
    corrected = None
    integer_only = config.get('integer_only', False)
    
    try:
        val = float(str(raw_value).replace('%', '').strip())
        
        # For integer-only values, convert to int
        if integer_only:
            val = int(val)
        
        if config['min'] <= val <= config['max']:
            corrected = str(int(val)) if integer_only else str(val)
        else:
            issues.append(f"Out of range: {val} not in [{config['min']}, {config['max']}]")
    except ValueError:
        issues.append(f"Cannot parse as number: '{raw_value}'")
    
    # If initial parse failed or out of range, try fixing decimal
    if corrected is None:
        fixed = try_fix_decimal(
            str(raw_value),
            config['decimals'],
            config['min'],
            config['max'],
            integer_only=integer_only
        )
        if fixed:
            corrected = fixed
            correction_msg = (
                f"Removed decimal from integer value: {raw_value} → {fixed}" 
                if integer_only and '.' in str(raw_value) 
                else f"Applied decimal correction: {raw_value} → {fixed}"
            )
            issues.append(correction_msg)
    
    return {
        'valid': corrected is not None,
        'corrected_value': corrected,
        'raw_value': raw_value,
        'issues': issues if issues else None
    }


def apply_postprocessing(predictions, min_confidence=0.5, previous_valid=None, use_previous_on_invalid=False):
    """
    Apply validation and corrections to OCR predictions.
    
    Args:
        predictions: dict of class_name -> {value, det_conf, ocr_conf, bbox}
        min_confidence: minimum OCR confidence threshold
        previous_valid: dict of class_name -> last valid value (for live video smoothing)
        use_previous_on_invalid: If True, use previous valid value when current is invalid
                                 (for live video streams only)
        
    Returns:
        dict with:
            - 'corrected_predictions': dict of valid predictions
            - 'validation_log': dict of validation status per class
            - 'previous_valid': updated dict of last valid values (if provided)
    """
    if previous_valid is None:
        previous_valid = {}
    
    corrected = {}
    validation_log = {}
    
    for class_name in NUMERIC_CLASSES:
        if class_name not in predictions:
            # No detection for this class
            validation_log[class_name] = {
                'status': 'not_detected',
                'issues': ['No detection for this parameter'],
                'used_previous': False
            }
            
            # For live video, use previous valid value if available
            if use_previous_on_invalid and class_name in previous_valid:
                corrected[class_name] = {
                    'value': previous_valid[class_name],
                    'source': 'previous_valid',
                    'det_conf': None,
                    'ocr_conf': None
                }
                validation_log[class_name]['used_previous'] = True
            continue
        
        pred_data = predictions[class_name]
        raw_value = pred_data.get('value')
        ocr_conf = pred_data.get('ocr_conf', 0.0)
        det_conf = pred_data.get('det_conf', 0.0)
        
        # Validate and correct
        result = validate_and_correct_value(
            class_name,
            raw_value,
            ocr_conf,
            min_confidence
        )
        
        if result['valid']:
            # Use corrected value
            corrected[class_name] = {
                'value': result['corrected_value'],
                'raw_value': result['raw_value'],
                'source': 'ocr_corrected' if result['issues'] else 'ocr_direct',
                'det_conf': det_conf,
                'ocr_conf': ocr_conf,
                'bbox': pred_data.get('bbox'),
                'status': 'valid'
            }
            
            # Update previous valid value
            previous_valid[class_name] = result['corrected_value']
            
            validation_log[class_name] = {
                'status': 'valid',
                'issues': result['issues'],
                'used_previous': False
            }
        else:
            # Invalid value
            validation_log[class_name] = {
                'status': 'invalid',
                'issues': result['issues'],
                'used_previous': False
            }
            
            # For live video, use previous valid value if available
            if use_previous_on_invalid and class_name in previous_valid:
                corrected[class_name] = {
                    'value': previous_valid[class_name],
                    'source': 'previous_valid',
                    'det_conf': None,
                    'ocr_conf': None,
                    'invalid_raw': result['raw_value'],
                    'status': 'carried_forward'
                }
                validation_log[class_name]['used_previous'] = True
            # Otherwise, don't add to corrected dict - parameter will show as None/missing
    
    return {
        'corrected_predictions': corrected,
        'validation_log': validation_log,
        'previous_valid': previous_valid
    }


def format_display_value(class_name, corrected_data):
    """
    Format a corrected value for display with unit.
    
    Args:
        class_name: Name of the parameter class
        corrected_data: dict from corrected_predictions
        
    Returns:
        Formatted string for display (e.g., "145 bpm", "36.5 °C")
    """
    if class_name not in VALUE_RANGES:
        return "N/A"
    
    if corrected_data is None:
        return "N/A"
    
    config = VALUE_RANGES[class_name]
    value = corrected_data.get('value')
    
    if value is None:
        return "N/A"
    
    return f"{value} {config['unit']}"


def get_validation_status_emoji(status):
    """
    Get emoji indicator for validation status.
    
    Args:
        status: Validation status string
        
    Returns:
        Emoji string
    """
    status_map = {
        'valid': '✅',
        'invalid': '❌',
        'not_detected': '⚠️'
    }
    return status_map.get(status, '❓')
